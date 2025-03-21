import ldm_patched.modules.model_base
import logging
import torch

class NRS:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "squash": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                              "stretch": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 30.0, "step": 0.01}),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "advanced/model"

    def patch(self, model, squash, stretch):
        def nrs(args):
            cond = args["cond"]
            uncond = args["uncond"]
            cond_scale = args["cond_scale"]
            sigma = args["sigma"]
            sigma = sigma.view(sigma.shape[:1] + (1,) * (cond.ndim - 1))
            x_orig = args["input"]

            logging.debug(f"NRS.nrs: CFG: {cond_scale}, Squash: {squash}, Stretch: {stretch}")

            #rescale cfg has to be done on v-pred model output
            x = x_orig / (sigma * sigma + 1.0)
            cond = ((x - (x_orig - cond)) * (sigma ** 2 + 1.0) ** 0.5) / (sigma)
            uncond = ((x - (x_orig - uncond)) * (sigma ** 2 + 1.0) ** 0.5) / (sigma)
            logging.debug(f"NRS.nrs: generated cond and uncond")

            x_final = None
            if False:
                # displace cond by rejection of uncond on cond
                u_dot_c = torch.sum(uncond * cond, dim=-1, keepdim=True)
                c_dot_c = torch.sum(cond * cond, dim=-1, keepdim=True)
                u_on_c = (u_dot_c / c_dot_c) * cond
                u_rej_c = uncond - u_on_c
                displaced = (cond - cond_scale * u_rej_c)
                logging.debug(f"NRS.nrs: displaced")

                # squash displaced vector towards len(cond) based on squash scale
                sq_len = torch.sum(displaced * displaced, dim=-1, keepdim=True)
                squash_scale = (1 - squash) + squash * ((c_dot_c/sq_len) ** 0.5)
                squashed = displaced * squash_scale
                logging.debug(f"NRS.nrs: squashed")

                # stretch turned vector towards cond based on stretch scale
                sq_dot_c = torch.sum(squashed * cond, dim=-1, keepdim=True)
                sq_on_c = (sq_dot_c / c_dot_c) * cond
                x_final = squashed + sq_on_c * stretch
                logging.debug(f"NRS.nrs: final")
            else:
                # displace cond by rejection of uncond on cond
                u_dot_c = torch.sum(uncond * cond, dim=-1, keepdim=True)
                c_dot_c = torch.sum(cond * cond, dim=-1, keepdim=True)
                u_on_c = (u_dot_c / c_dot_c) * cond
                u_rej_c = uncond - u_on_c
                displaced = cond + stretch * (cond - torch.clamp(u_dot_c / c_dot_c, min=0, max=1) * cond) - cond_scale * u_rej_c
                logging.debug(f"NRS.nrs: displaced & stretched")

                # squash displaced vector towards len(cond) based on squash scale
                sq_len = torch.sum(displaced * displaced, dim=-1, keepdim=True)
                squash_scale = (1 - squash) + squash * ((c_dot_c/sq_len) ** 0.5)
                x_final = displaced * squash_scale
                logging.debug(f"NRS.nrs: final")

            return x_orig - (x - x_final * sigma / (sigma * sigma + 1.0) ** 0.5)
        
        m = model.clone()
        m.set_model_sampler_cfg_function(nrs, True)
        return (m, )

NODE_CLASS_MAPPINGS = {
    "NRS": NRS,
}