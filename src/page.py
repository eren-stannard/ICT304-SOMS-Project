# page.py

"""
    Smart Occupancy Monitoring System - Client Application
    
    Authors:
        Ervin Galas    (34276705)
        Sofia Peeva    (35133522)
        Eren Stannard  (34189185)
    
    ICT304: AI System Design
    Murdoch University
    
    Purpose of File:
    Streamlit page creation.

"""


# Libraries used
import streamlit as st
from streamlit.navigation.page import StreamlitPage


class Page(StreamlitPage):
    """
    Streamlit app page based on `StreamlitPage`.
    
    Attributes
    ----------
    icon : str | None, optional, default=None
        Icon to use for page.
    page_url : str
        Path to page file.
    subtitle : str | None, optional, default=None
        Optional subtitle of page to use as header.
    title : str | None, optional, default=None
        Title of page.
    
    See Also
    --------
    StreamlitPage : Streamlit app page base class.
        [[source]](https://github.com/streamlit/streamlit/blob/1.45.0/lib/streamlit/navigation/page.py#L128)
    """
    
    def __init__(
        self, page: str, title: str | None = None, short_title: str | None = None, icon: str | None = None, default: bool = False,
    ) -> None:
        """
        Page class constructor.
        
        Parameters
        ----------
        page : str
            Path to page file.
        title : str | None, optional, default=None
            Title of page.
        short_title : str | None, optional, default=None
            Optional short/abbreviated title of page to use.
        icon : str | None, optional, default=None
            Icon to use for page.
        default : bool, optional, default=False
            If True, launch page at startup.
        """
        
        super(Page, self).__init__(
            page=page,
            title=title,
            icon=icon,
            default=default,
        )
        
        self.page = page
        self.short_title = short_title
                
        return
    
    
    def header(self) -> None:
        """Display page title and header in `create_page` method."""
        
        if self.short_title:
            st.title(f":primary[{self.icon}] {self.short_title}")
            st.header(self.title, divider='grey')
        else:
            st.title(f":primary[{self.icon}] {self.title}")
        
        return


def create_page(
    page: str, title: str | None = None, short_title: str | None = None, icon: str | None = None, default: bool = False,
) -> Page:
    """
    Create a new Streamlit page.

    Parameters
    ----------
    page : str
        Path to page file.
    title : str | None, optional, default=None
        Title of page.
    short_title : str | None, optional, default=None
        Optional short/abbreviated title of page to use.
    icon : str | None, optional, default=None
        Icon to use for page.
    default : bool, optional, default=False
        If True, launch page at startup.
    
    Returns
    -------
    page : Page
        New Streamlit page.
    
    See Also
    --------
    st.Page : Initialise `StreamlitPage` object.
        [[source]](https://github.com/streamlit/streamlit/blob/1.45.0/lib/streamlit/navigation/page.py#L29)
    """
    
    return Page(
        page=page,
        title=title,
        short_title=short_title,
        icon=icon,
        default=default,
    )