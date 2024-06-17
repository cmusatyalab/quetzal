# import streamlit.components.v1 as components
# from pathlib import Path

# def image_frame(image_urls=[], 
#                 captions=[],
#                 label="",
#                 starting_point=30,
#                 border=True, 
#                 padding="0px 8px", 
#                 font_family="sans-serif", 
#                 card_border_radius="0.7rem", 
#                 image_border_radius="0.65rem",
#                 dark_mode=False,
#                 key=None):
#     _image_frame = components.declare_component(
#         name='image_frame',
#         path=str(Path(__file__).parent)
#     )

#     # Path(__file__).parent.joinpath("app.py")
#     if border:
#         border_width = "1px" 
#     else:
#         border_width = "0px"
#         card_border_radius = "0rem"
#         image_border_radius = "0rem",
    
#     # Pass parameters to the frontend
#     return _image_frame(
#         image_urls=image_urls, 
#         captions=captions, 
#         label=label,
#         border_width = border_width,
#         starting_point=starting_point, 
#         padding=padding, 
#         font_family=font_family, 
#         card_border_radius=card_border_radius, 
#         image_border_radius=image_border_radius,
#         dark_mode=dark_mode,
#         key=key
#     )

#quetzal/quetzal_app/elements/image_frame_component/__init__.py
import streamlit.components.v1 as components
from pathlib import Path

def image_frame(image_urls=[[]], 
                captions=[[]],
                labels=[[]],
                starting_point=30,
                border=True, 
                padding="0px 8px", 
                font_family="sans-serif", 
                card_border_radius="0.7rem", 
                image_border_radius="0.65rem",
                dark_mode=False,
                key=None):

    _image_frame = components.declare_component(
        name='image_frame',
        path=str(Path(__file__).parent)
    )

    # Path(__file__).parent.joinpath("app.py")
    if border:
        border_width = "1px" 
    else:
        border_width = "0px"
        card_border_radius = "0rem"
        image_border_radius = "0rem",
    
    # Pass parameters to the frontend
    return _image_frame(
        image_urls=image_urls, 
        captions=captions, 
        labels=labels,
        border_width = border_width,
        starting_point=starting_point, 
        padding=padding, 
        font_family=font_family, 
        card_border_radius=card_border_radius, 
        image_border_radius=image_border_radius,
        dark_mode=dark_mode,
        key=key
    )
