import gradio as gr

temp = 0
def inc():
    global temp
    temp = temp + 1
    return temp

def main():

    with gr.Blocks() as demo:
        slider = gr.Slider(0, 100, 10, interactive=True)
        play_btn = gr.Button("►")
        # stop_btn = gr.Button("Stop")
        run_state = gr.State(False)
        slider_idx = gr.Number(0, visible=False)

        def inc_local():
            if run_state.value:
                slider.value = slider.value + 1
                return slider.value
            return gr.update()
        
        def toggle_run(play_btn):
            if play_btn == "►":
                run_state.value = True
                return "Stop AutoPlay"
            else:
                run_state.value = False
                return "►"
            
        def set_sequence(idx):
            slider.value = idx
                    
        play_btn.click()
        demo.load(inc_local, inputs=None, outputs=slider_idx, every=1)
        slider_idx.change(lambda x:x, inputs=slider_idx, outputs=slider, show_progress=False)
        slider.release(set_sequence, inputs=slider, outputs=None, show_progress=False)
        play_btn.click(toggle_run, inputs=play_btn, outputs=play_btn)

    demo.queue()
    demo.launch()


if __name__ == "__main__":
    main()

# pip install markupsafe==2.0.1


# https://www.gradio.app/guides/running-gradio-on-your-web-server-with-nginx
