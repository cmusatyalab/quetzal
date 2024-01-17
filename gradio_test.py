# import gradio as gr

# temp = 0
# def inc():
#     global temp
#     temp = temp + 1
#     return temp


# import time
# import datetime

# def main():

#     with gr.Blocks() as demo:
#         slider = gr.Slider(0, 100, 10, interactive=True)
#         play_btn = gr.Button("►",)
#         run_state = gr.State(False)
#         wakeup_time = gr.State(datetime.datetime.now())
#         slider_idx = gr.Number(0, visible=False)
#         # stop

        
#         def toggle_run(play_btn, slider):
#             if play_btn == "►":
#                 return "Stop AutoPlay", True, slider + 1, datetime.datetime.now() + datetime.timedelta(seconds=0.5)
#             else:
#                 return "►", False, slider, datetime.datetime.now()
            
#         def update_slider(val):
#             return val
            
#         def update_slider_idx(val, run_state, wake_up_time):
#             if run_state:
#                 time_to_sleep = (wake_up_time - datetime.datetime.now()).total_seconds()
#                 time.sleep(max(time_to_sleep, 0))
#                 return val + 1, datetime.datetime.now() + datetime.timedelta(seconds=0.5), gr.update()
#             elif val == slider.maximum:
#                 return val, datetime.datetime.now(), gr.update()
#             return val, datetime.datetime.now(), False
        
#         # demo.load(inc_local, inputs=run_state, outputs=slider_idx, every=1)
#         slider_idx.change(update_slider_idx, inputs=[slider_idx, run_state, wakeup_time], outputs=[slider, wakeup_time, run_state], show_progress=False).then(update_slider, inputs=slider, outputs=slider_idx)
#         play_btn.click(toggle_run, inputs=[play_btn, slider], outputs=[play_btn, run_state, slider, wakeup_time], show_progress=False).then(update_slider, inputs=slider, outputs=slider_idx)

#     demo.queue()
#     demo.launch()


# if __name__ == "__main__":
#     main()

# # pip install markupsafe==2.0.1


# # https://www.gradio.app/guides/running-gradio-on-your-web-server-with-nginx
    

# # example of gradio mount on fastapi # https://github.com/gradio-app/gradio/issues/2654
import gradio as gr

def toggle_view(show_first_page):
    
    return gr.update(visible=show_first_page), gr.update(visible= not show_first_page), not show_first_page

with gr.Blocks() as app:
    with gr.Row():
        show_first_page = gr.State(True)
        toggle_button = gr.Button("Toggle View")

    with gr.Column(visible=True) as first_column:
        gr.Markdown("### First Page Content Here")

    with gr.Column(visible=False) as second_column:
        gr.Markdown("### Second Page Content Here")

    toggle_button.click(toggle_view, inputs=[show_first_page], outputs=[first_column, second_column, show_first_page])

# Launch the Gradio app
app.launch()