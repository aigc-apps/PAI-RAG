from typing import Dict, Any
import gradio as gr


def create_evaluation_tab() -> Dict[str, Any]:
    with gr.Row():
        with gr.Column():
            agent_type = gr.Radio(
                ["FunctionCalling", "ReACT"],
                label="\N{fire} Agent Type",
                elem_id="agent_type",
                value="FunctionCalling",
            )

        return {agent_type.elem_id: agent_type}
