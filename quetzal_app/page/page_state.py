from abc import ABC, abstractmethod

DEFAULT_BOX_TH = 0.3
DEFAULT_TEXT_TH = 0.25
DEFAULT_SLIDER_VAL = 0      
    
class PageState:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        
    def __getitem__(self, key):
        return getattr(self, key, None)

    def __setitem__(self, key, value):
        setattr(self, key, value)
        
    def update(self, updates: dict):
        for key, value in updates.items():
            setattr(self, key, value)
            
class AppState:
    def __init__(self):
        self._pages = {}
        
    def __getitem__(self, key):
        return getattr(self, key, None)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __getattr__(self, page_name):
        if page_name not in self._pages:
            self._pages[page_name] = PageState()
        return self._pages[page_name]

    def __setattr__(self, page_name, value):
        if page_name == "_pages":
            super().__setattr__(page_name, value)
        else:
            self._pages[page_name] = value

    def initialize_page(self, page_name, **kwargs):
        self._pages[page_name] = PageState(**kwargs)

class Page(ABC):
    name = "default_page"
    state = None
    
    @abstractmethod
    def __init__(self, root_state: PageState, to_page: list[callable]):
        pass
    
    # @staticmethod
    @abstractmethod
    def render(self):
        pass
    
    # @staticmethod
    @abstractmethod
    def init_page_state(self, root_state: PageState) -> PageState:
        pass


# # Example usage
# page_state = AppState()

# # Initialize page state with predefined keys and values
# page_state.initialize_page('home', title='Home Page', content='Welcome to the Home Page!')

# # Access a page's state and read a value
# print(page_state.home.title)  # Output: Home Page

# # Set a new value for a key in a page's state
# page_state.home.title = 'Updated Home Page'
# print(page_state.home.title)  # Output: Updated Home Page

# # Accessing and setting values for another page
# page_state.about.description = 'About our company'
# print(page_state.about.description)  # Output: About our company
