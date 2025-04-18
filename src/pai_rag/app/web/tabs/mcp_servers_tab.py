import gradio as gr
import pandas as pd
from pai_rag.app.web.utils import components_to_dict
import pai_rag.app.web.event_listeners as ev_listeners
from pai_rag.app.web.rag_local_client import rag_client


def create_mcp_servers_tab():
    rag_config = rag_client.get_config()
    with gr.Row():
        with gr.Column(scale=1):
            _ = gr.Markdown(value="## \N{WHITE MEDIUM STAR} **配置 MCP 服务**")
            with gr.Column(variant="panel"):
                # mcp选择区域
                mcp_server_choices = ["NEW"] + [
                    mcp_server.name
                    for mcp_server in rag_config.mcp_servers
                    if mcp_server.name
                ]

                with gr.Row():
                    mcp_servers = gr.Dropdown(
                        label="MCP配置",
                        choices=mcp_server_choices,
                        value="NEW",
                        interactive=True,
                        elem_id="mcp_servers",
                        allow_custom_value=False,
                    )

            with gr.Row(visible=True) as config_row:
                with gr.Column():
                    with gr.Row():
                        mcp_server_name = gr.Textbox(
                            label="MCP 服务名称",
                            elem_id="mcp_server_name",
                            interactive=True,
                        )
                        mcp_server_url = gr.Textbox(
                            label="MCP 服务URL",
                            elem_id="mcp_server_url",
                            interactive=True,
                        )
                        transport_type = gr.Dropdown(
                            label="传输方式",
                            choices=["sse"],
                            value="sse",
                            interactive=True,
                            elem_id="transport_type",
                            allow_custom_value=False,
                        )
                    with gr.Row():
                        active_status = gr.Checkbox(
                            label="激活",
                            value=False,
                            interactive=True,
                            elem_id="active_status",
                            container=True,
                            scale=1,
                        )
            with gr.Column():
                save_mcp_server_btn = gr.Button(
                    value="添加MCP Server",
                    elem_id="save_mcp_server_btn",
                    variant="primary",
                )
            with gr.Column():
                delete_btn = gr.Button(
                    "删除MCP Server",
                    visible=bool(rag_config.mcp_servers),
                    variant="secondary",
                )

            mcp_servers.change(
                fn=ev_listeners.update_mcp_servers,
                inputs=mcp_servers,
                outputs=[
                    config_row,
                    save_mcp_server_btn,
                    delete_btn,
                    mcp_server_name,
                    mcp_server_url,
                    transport_type,
                    active_status,
                ],
            )

        with gr.Column(scale=4):
            _ = gr.Markdown(value="## \N{WHITE MEDIUM STAR} **MCP Server**")
            with gr.Row():
                update_mcp_info = gr.Button(
                    value="刷新MCP服务状态",
                    elem_id="update_mcp_info",
                    variant="primary",
                )
            with gr.Row():
                mcp_servers_data = [
                    {
                        "是否激活": "yes" if mcp_server.activated else "no",
                        "MCP Server名称": mcp_server.name,
                        "MCP Server URL": mcp_server.url,
                        "MCP Server Transport": mcp_server.transport,
                        "MCP Server Description": mcp_server.description,
                    }
                    for mcp_server in rag_config.mcp_servers
                ]
                mcp_servers_data = pd.DataFrame(mcp_servers_data)

                mcp_servers_display = gr.DataFrame(
                    label="",
                    visible=True,
                    elem_id="mcp_servers_display",
                    headers=[
                        "是否激活",
                        "MCP Server名称",
                        "MCP Server URL",
                        "MCP Server Transport",
                        "MCP Server Description",
                    ],
                    value=mcp_servers_data,
                )
            # 绑定删除按钮的事件
            delete_btn.click(
                fn=ev_listeners.delete_mcp_server,
                inputs=mcp_servers,
                outputs=[mcp_servers, mcp_servers_display, config_row, delete_btn],
            )

            save_mcp_server_btn.click(
                fn=ev_listeners.save_new_mcp_server,
                inputs=[
                    mcp_servers,
                    mcp_server_name,
                    mcp_server_url,
                    transport_type,
                    active_status,
                ],
                outputs=[
                    mcp_servers,
                    mcp_servers_display,
                    save_mcp_server_btn,
                    delete_btn,
                ],
            )
            update_mcp_info.click(
                fn=ev_listeners.list_mcp_servers,
                outputs=mcp_servers_display,
            )

    components = [mcp_servers, mcp_servers_display]
    elems = components_to_dict(components)
    return elems
