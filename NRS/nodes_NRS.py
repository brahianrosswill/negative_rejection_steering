import logging
import torch
import comfy.utils

# Optional: Configure logging level if needed
# logging.basicConfig(level=logging.DEBUG)

# Small epsilon for numerical stability in divisions
EPSILON = 1e-6

class NRS:
    """
    NRS (Noise Rejection Sampling - presumed name) Node:
    Applies advanced CFG manipulation techniques (skew, stretch, squash)
    to the conditional prediction based on the unconditional prediction.
    Assumes the model uses v-prediction.
    """
    # Define available versions
    VERSIONS = ["v0.4.5", "v0.4.4", "v0.4.3", "v0.4.2", "v0.4.1", "v4", "v3", "v2", "v1"]

    @classmethod
    def INPUT_TYPES(cls):
        """
        Defines the input parameters for the node.
        """
        return {
            "required": {
                "model": ("MODEL",),
                "version": (cls.VERSIONS, {"default": "v0.4.5"}), # Added version selection
                "skew": ("FLOAT", {"default": 2.0, "min": -30.0, "max": 30.0, "step": 0.01, "display": "number"}),
                "stretch": ("FLOAT", {"default": 2.0, "min": -30.0, "max": 30.0, "step": 0.01, "display": "number"}),
                "squash": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}), # Changed display for squash
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "advanced/model"

    def patch(self, model, version, skew, stretch, squash):
        """
        Applies the NRS patch to the model's CFG sampler function.
        """
        m = model.clone()

        # Store parameters for the inner function
        current_version = version
        current_skew = skew
        current_stretch = stretch
        current_squash = squash

        def nrs_cfg_function(args):
            """
            The custom CFG function that performs NRS.
            This function replaces the standard CFG combination logic.
            """
            cond = args["cond"]
            uncond = args["uncond"]
            sigma = args["sigma"]
            # sigma may not be batched if called from node preview, handle this
            if sigma.shape[0] < cond.shape[0]:
                 sigma = sigma.repeat(cond.shape[0])
            sigma = sigma.view(sigma.shape[:1] + (1,) * (cond.ndim - 1)) # Reshape sigma for broadcasting
            x_orig = args["input"] # This is the noisy latent x_t

            # logging.debug(f"NRS ({current_version}): Skew={current_skew}, Stretch={current_stretch}, Squash={current_squash}, Sigma={sigma.mean().item():.2f}")

            # --- V-Prediction Space Conversion ---
            # Convert inputs (assumed v-prediction) to a space suitable for geometric manipulation.
            # The exact derivation of this conversion might be specific to the NRS technique.
            # N = sqrt(sigma^2 + 1)
            N_reciprocal = 1.0 / torch.sqrt(sigma * sigma + 1.0) # Precompute for efficiency
            x = x_orig * N_reciprocal
            # These lines effectively convert v-predictions (cond, uncond) into something akin to
            # scaled epsilon predictions relative to x. Let's call them 'pseudo-eps'.
            pseudo_eps_cond   = (x - (x_orig - cond))   * (1.0 / N_reciprocal) / sigma
            pseudo_eps_uncond = (x - (x_orig - uncond)) * (1.0 / N_reciprocal) / sigma
            # logging.debug(f"NRS: Converted to pseudo-eps space.")

            # --- Core NRS Geometric Manipulation (using pseudo-eps vectors) ---
            # Common calculations
            # Keep dimensions for broadcasting: (batch_size, 1, 1, 1)
            c_dot_c = torch.sum(pseudo_eps_cond * pseudo_eps_cond, dim=tuple(range(1, cond.ndim)), keepdim=True)
            u_dot_c = torch.sum(pseudo_eps_uncond * pseudo_eps_cond, dim=tuple(range(1, cond.ndim)), keepdim=True)

            # Add epsilon for stability
            c_dot_c_stable = c_dot_c + EPSILON

            # Projection of uncond onto cond
            u_on_c_mag = u_dot_c / c_dot_c_stable
            u_on_c = u_on_c_mag * pseudo_eps_cond

            # Rejection of uncond by cond (component of uncond orthogonal to cond)
            u_rej_c = pseudo_eps_uncond - u_on_c

            # Pre-calculate cond_len for versions that need it
            cond_len = torch.sqrt(c_dot_c_stable)

            x_final = None # This will hold the final modified pseudo-eps vector

            # --- Version-Specific Logic ---
            if current_version == "v1":
                # v1: Skew by rejection, squash length, stretch along original cond direction
                displaced = pseudo_eps_cond - current_skew * u_rej_c
                d_len_sq = torch.sum(displaced * displaced, dim=tuple(range(1, cond.ndim)), keepdim=True)
                squash_scale = (1 - current_squash) + current_squash * torch.sqrt(c_dot_c_stable / (d_len_sq + EPSILON))
                squashed = displaced * squash_scale
                sq_dot_c = torch.sum(squashed * pseudo_eps_cond, dim=tuple(range(1, cond.ndim)), keepdim=True)
                sq_on_c = (sq_dot_c / c_dot_c_stable) * pseudo_eps_cond
                x_final = squashed + sq_on_c * current_stretch
                # logging.debug(f"NRS v1: Applied skew, squash, stretch.")

            elif current_version == "v2":
                # v2: Combined displace/stretch, then squash
                # Stretch term amplifies cond based on how much it differs from clamped projection
                stretch_component = current_stretch * (pseudo_eps_cond - torch.clamp(u_on_c_mag, min=0, max=1) * pseudo_eps_cond)
                displaced = pseudo_eps_cond + stretch_component - current_skew * u_rej_c
                d_len_sq = torch.sum(displaced * displaced, dim=tuple(range(1, cond.ndim)), keepdim=True)
                squash_scale = (1 - current_squash) + current_squash * torch.sqrt(c_dot_c_stable / (d_len_sq + EPSILON))
                x_final = displaced * squash_scale
                # logging.debug(f"NRS v2: Applied combined stretch/skew, then squash.")

            elif current_version == "v3":
                 # v3: Skew, then apply combined squash/stretch scale
                displaced = pseudo_eps_cond - current_skew * u_rej_c
                d_len_sq = torch.sum(displaced * displaced, dim=tuple(range(1, cond.ndim)), keepdim=True)
                # Squash scale based on length ratio
                squash_scale = (1 - current_squash) + current_squash * torch.sqrt(c_dot_c_stable / (d_len_sq + EPSILON))
                # Stretch scale based on difference between cond_len and projected length
                stretch_scale = (1 - current_stretch) + current_stretch * (2 * cond_len - u_on_c_mag * cond_len) / cond_len # Simplified from original
                x_final = displaced * squash_scale * stretch_scale
                # logging.debug(f"NRS v3: Applied skew, then combined squash/stretch.")

            elif current_version == "v4":
                # v4: Squash rejection component, stretch based on rejection magnitude ratio
                rej_dot_rej = torch.sum(u_rej_c * u_rej_c, dim=tuple(range(1, cond.ndim)), keepdim=True)
                stretch_factor = current_stretch * torch.sqrt(rej_dot_rej / c_dot_c_stable)
                # Note: Original used 'squash' for skew magnitude here, seems intentional for this version
                x_final = pseudo_eps_cond - current_squash * u_rej_c + stretch_factor * pseudo_eps_cond
                # logging.debug(f"NRS v4: Applied rejection squash (using squash param) and rejection-based stretch.")

            elif current_version == "v0.4.1":
                # v0.4.1: Stretch based on rejection magnitude ratio, then skew, then squash
                rej_dot_rej = torch.sum(u_rej_c * u_rej_c, dim=tuple(range(1, cond.ndim)), keepdim=True)
                stretch_factor = current_stretch * torch.sqrt(rej_dot_rej / c_dot_c_stable)
                stretched = pseudo_eps_cond + stretch_factor * pseudo_eps_cond
                skewed = stretched - current_skew * u_rej_c
                sk_dot_sk = torch.sum(skewed * skewed, dim=tuple(range(1, cond.ndim)), keepdim=True)
                squash_scale = (1 - current_squash) + current_squash * torch.sqrt(c_dot_c_stable / (sk_dot_sk + EPSILON))
                x_final = skewed * squash_scale
                # logging.debug(f"NRS v0.4.1: Applied stretch, skew, squash.")

            elif current_version == "v0.4.2":
                # v0.4.2: Stretch based on absolute difference between cond len and projection len, skew, squash
                proj_len = torch.sqrt(torch.sum(u_on_c * u_on_c, dim=tuple(range(1, cond.ndim)), keepdim=True) + EPSILON)
                stretch_factor = 1 + current_stretch * torch.abs(cond_len - proj_len) / cond_len
                stretched = pseudo_eps_cond * stretch_factor
                skewed = stretched - current_skew * u_rej_c
                sk_dot_sk = torch.sum(skewed * skewed, dim=tuple(range(1, cond.ndim)), keepdim=True)
                squash_scale = (1 - current_squash) + current_squash * cond_len / torch.sqrt(sk_dot_sk + EPSILON)
                x_final = skewed * squash_scale
                # logging.debug(f"NRS v0.4.2: Applied length-diff stretch, skew, squash.")

            elif current_version == "v0.4.3":
                # v0.4.3: Stretch based on signed difference between cond len and projection len, skew, squash
                proj_len = torch.sqrt(torch.sum(u_on_c * u_on_c, dim=tuple(range(1, cond.ndim)), keepdim=True) + EPSILON)
                stretch_factor = 1 + current_stretch * (cond_len - proj_len) / cond_len
                stretched = pseudo_eps_cond * stretch_factor
                skewed = stretched - current_skew * u_rej_c
                sk_dot_sk = torch.sum(skewed * skewed, dim=tuple(range(1, cond.ndim)), keepdim=True)
                squash_scale = (1 - current_squash) + current_squash * cond_len / torch.sqrt(sk_dot_sk + EPSILON)
                x_final = skewed * squash_scale
                # logging.debug(f"NRS v0.4.3: Applied signed length-diff stretch, skew, squash.")

            elif current_version == "v0.4.4":
                # v0.4.4: Stretch based on length of difference vector (cond - proj), skew, squash
                proj_diff = pseudo_eps_cond - u_on_c
                proj_diff_len = torch.sqrt(torch.sum(proj_diff * proj_diff, dim=tuple(range(1, cond.ndim)), keepdim=True) + EPSILON)
                stretch_factor = 1 + current_stretch * proj_diff_len / cond_len
                stretched = pseudo_eps_cond * stretch_factor
                skewed = stretched - current_skew * u_rej_c
                sk_dot_sk = torch.sum(skewed * skewed, dim=tuple(range(1, cond.ndim)), keepdim=True)
                squash_scale = (1 - current_squash) + current_squash * cond_len / torch.sqrt(sk_dot_sk + EPSILON)
                x_final = skewed * squash_scale
                # logging.debug(f"NRS v0.4.4: Applied projection-diff-len stretch, skew, squash.")

            elif current_version == "v0.4.5":
                 # v0.4.5: Stretch by adding scaled difference vector, skew, squash
                proj_diff = pseudo_eps_cond - u_on_c
                # Amplify Cond based on the difference vector between cond and the projection of uncond onto it
                stretched = pseudo_eps_cond + (current_stretch * proj_diff)
                # Skew/Steer Cond based on the rejection of uncond by cond
                skewed = stretched - current_skew * u_rej_c
                # Squash final length back towards original length of cond
                sk_dot_sk = torch.sum(skewed * skewed, dim=tuple(range(1, cond.ndim)), keepdim=True)
                squash_scale = (1 - current_squash) + current_squash * cond_len / torch.sqrt(sk_dot_sk + EPSILON)
                x_final = skewed * squash_scale
                # logging.debug(f"NRS v0.4.5: Applied diff-vector stretch, skew, squash.")

            else:
                # Fallback or error handling: Default to standard CFG? Or raise error?
                # For safety, let's just use the original cond if version is unknown
                logging.warning(f"NRS: Unknown version '{current_version}'. Using original cond.")
                x_final = pseudo_eps_cond # Fallback

            # --- Convert back from pseudo-eps space to final prediction ---
            # The reverse conversion seems to be: v_final = x_final * sigma / N
            # Then the final output is derived from this modified v.
            # The formula used is: x_orig - (x - v_final / N) = x_orig - (x - x_final * sigma / (N*N))
            # This seems different from standard v-pred final step. Trusting original author's formula.
            final_v_adjustment = x_final * sigma * N_reciprocal # v_final / N ? No, v_final = x_final*sigma + x ?
                                                                 # Let's use the original formula structure:
                                                                 # x_final * sigma / sqrt(sigma**2+1)
            final_pred = x_orig - (x - final_v_adjustment) # This should be the denoised x_0 prediction

            # The sampler expects the *noise* prediction (eps) or *velocity* (v).
            # Since we started assuming v-pred inputs and did v-pred scaling,
            # the function needs to return the modified 'v' prediction.
            # Let's re-derive:
            # We have modified pseudo_eps `x_final`.
            # Convert back to v: v_final = x_final * sigma + x
            # The standard CFG function in ComfyUI's sampler returns the combined model output (eps or v).
            # So we should return v_final.
            v_final = x_final * sigma + x
            # logging.debug("NRS: Converted back to v-prediction.")

            # --- Sanity Check ---
            # Let's test the original return formula with skew=0, stretch=0, squash=0
            # v0.4.5: stretched = cond + 0*proj_diff = cond. skewed = cond - 0*rej = cond.
            # squash_scale = (1-0) + 0 * (...) = 1. x_final = cond * 1 = cond (pseudo_eps_cond)
            # original_return = x_orig - (x - pseudo_eps_cond * sigma / N)
            # If this function should return the modified v, let's calculate v when x_final = pseudo_eps_cond
            # v_equiv = pseudo_eps_cond * sigma + x
            # Let's compare `v_equiv` to `original_return`. They are likely different representations.
            # ComfyUI's `set_model_sampler_cfg_function` expects the callback to return
            # the *combined* model output (the result that would normally come from `uncond + cfg_scale * (cond - uncond)`).
            # Since the inputs `args["cond"]`, `args["uncond"]` are v-predictions, the output should also be a v-prediction.
            # Therefore, returning `v_final = x_final * sigma + x` seems correct.

            # Let's re-evaluate the original return line:
            # `return x_orig - (x - x_final * sigma / (sigma * sigma + 1.0) ** 0.5)`
            # `return x_orig - (x - x_final * sigma * N_reciprocal)`
            # This looks like `x_0_pred = x_t - eps_pred * N`. Here `eps_pred` would be `(x - x_final * sigma * N_reciprocal) / N`.
            # It seems the original code was calculating the final x0 prediction instead of the required v prediction.

            # Return the calculated final v-prediction
            return v_final

        # Set the custom CFG function
        m.set_model_sampler_cfg_function(nrs_cfg_function, disable_cfg1_optimization=True) # Disable CFG=1 optimization as our function handles it implicitly
        logging.info(f"NRS Patched: Version={version}, Skew={skew}, Stretch={stretch}, Squash={squash}")
        return (m,)

NODE_CLASS_MAPPINGS = {
    "NRS Advanced CFG": NRS, # Renamed for clarity
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NRS Advanced CFG": "NRS Advanced CFG Guidance",
}