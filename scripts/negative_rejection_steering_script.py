import gradio as gr
import logging
import sys
from functools import partial
from modules import scripts, script_callbacks
from typing import Any

from NRS.nodes_NRS import NRS

class NRSScript(scripts.Script):
    def __init__(self):
        super().__init__()
        self.enabled = False
        self.squash = 0.5
        self.stretch = 1.0

    sorting_priority = 5

    def title(self):
        return "Negative Rejection Steering for reForge"

    def show(self, is_img2img):
        return scripts.AlwaysVisible
    
    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            gr.HTML("<p><i>Adjust the settings for Negative Rejection Steering.</i></p>")
            enabled = gr.Checkbox(label="Enable NRS", value=self.enabled)
            squash = gr.Slider(label="NRS Squash Multiplier", minimum=0.0, maximum=1.0, step=0.01, value=self.squash)
            stretch = gr.Slider(label="NRS Stretch Multiplier", minimum=-1.0, maximum=30.0, step=0.01, value=self.stretch)

        enabled.change(
            lambda x: self.update_enabled(x),
            inputs=[enabled]
        )

        return (enabled, squash, stretch)
    

    def update_enabled(self, value):
        self.enabled = value

    def process_before_every_sampling(self, p, *args, **kwargs):
        if len(args) >= 3:
            self.enabled, self.squash, self.stretch = args[:3]
        else:
            logging.warning("Not enough arguments provided to process_before_every_sampling")
            return

        xyz = getattr(p, "_nrs_xyz", {})
        if "enabled" in xyz:
            self.enabled = xyz["enabled"] == "True"
        if "squash" in xyz:
            self.squash = xyz["squash"]
        if "stretch" in xyz:
            self.stretch = xyz["stretch"]

        # Always start with a fresh clone of the original unet
        unet = p.sd_model.forge_objects.unet.clone()

        if not self.enabled:
            # Reset the unet to its original state
            p.sd_model.forge_objects.unet = unet
            return

        unet = NRS().patch(unet, self.squash, self.stretch)[0]

        p.sd_model.forge_objects.unet = unet
        p.extra_generation_params.update({
            "NRS_enabled": True,
            "NRS_squash": self.squash,
            "NRS_stretch": self.stretch,
        })

        logging.debug(f"NRS: Enabled: {self.enabled}, Squash: {self.squash}, Stretch: {self.stretch}")

        return

def set_value(p, x: Any, xs: Any, *, field: str):
    if not hasattr(p, "_nrs_xyz"):
        p._nrs_xyz = {}
    p._nrs_xyz[field] = x

def make_axis_on_xyz_grid():
    xyz_grid = None
    for script in scripts.scripts_data:
        if script.script_class.__module__ == "xyz_grid.py":
            xyz_grid = script.module
            break

    if xyz_grid is None:
        return

    axis = [
        xyz_grid.AxisOption(
            "(NRS) Enabled",
            str,
            partial(set_value, field="enabled"),
            choices=lambda: ["True", "False"]
        ),
        xyz_grid.AxisOption(
            "(NRS) Squash",
            float,
            partial(set_value, field="squash"),
        ),
        xyz_grid.AxisOption(
            "(NRS) Stretch",
            float,
            partial(set_value, field="stretch"),
        ),
    ]

    if not any(x.label.startswith("(NRS)") for x in xyz_grid.axis_options):
        xyz_grid.axis_options.extend(axis)

def on_before_ui():
    try:
        make_axis_on_xyz_grid()
    except Exception:
        error = traceback.format_exc()
        print(
            f"[-] NRS Script: xyz_grid error:\n{error}",
            file=sys.stderr,
        )

script_callbacks.on_before_ui(on_before_ui)
