"""
Streamlit Application: Interactive Softmax Visualization
=====================================================

This application provides an interactive demonstration of the softmax function,
allowing users to visualize how the function transforms input values and how
temperature affects the output distribution.

The softmax function (σ) converts a vector of real numbers into a probability
distribution, where all output values sum to 1.
"""

from typing import List, Optional
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray

# Set the page configuration with descriptive parameters
st.set_page_config(
    page_title="Softmax Interactive Demo",
    layout="centered",  # Centers content for better readability
    initial_sidebar_state="auto",  # Automatically show/hide sidebar
)

def create_sidebar_controls() -> tuple[NDArray[np.float64], float]:
    """
    Creates and manages all sidebar controls for the application.
    
    Returns:
        tuple: (input_numbers, temperature)
            - input_numbers: Array of random numbers to process
            - temperature: Temperature parameter for softmax calculation
    """
    with st.sidebar:
        st.title('Softmax Interactive Demo')
        
        # Display the mathematical formulas using LaTeX
        st.write("## Softmax Formula:")
        
        # Basic softmax formula
        st.latex(r"\sigma(z_i) = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}")
        
        # Temperature-scaled softmax formula
        st.write("### With Temperature:")
        st.latex(r"\sigma(z_i, T) = \frac{e^{z_i/T}}{\sum_{j=1}^K e^{z_j/T}}")
        
        st.write("""
        where:
        - z_i is the input value at position i
        - T is the temperature parameter
        - K is the number of input values
        """)

        # Initialize random numbers if they don't exist in session state
        if 'numbers' not in st.session_state:
            st.session_state.numbers = np.random.uniform(
                low=-1,
                high=2,
                size=10
            )

        # Button to generate new random numbers
        if st.button('Generate Random Numbers'):
            st.session_state.numbers = np.random.uniform(
                low=-1,
                high=2,
                size=10
            )

        # Temperature slider with fine-grained control
        temperature = st.slider(
            'Temperature',
            min_value=0.01,
            max_value=10.0,
            value=1.0,
            step=0.001,
            help="Controls the 'sharpness' of the distribution. Lower values make it more peaked."
        )
        
        return st.session_state.numbers, temperature

def softmax(arr: NDArray[np.float64], temperature: float) -> NDArray[np.float64]:
    """
    Computes the softmax function for the input array with temperature scaling.
    
    The softmax function normalizes an input vector into a probability distribution
    where all elements sum to 1. Temperature controls the "sharpness" of the distribution.
    
    Args:
        arr: Input array of real numbers
        temperature: Temperature parameter (T > 0)
            - High temperature (T → ∞) makes distribution more uniform
            - Low temperature (T → 0) makes distribution more peaked
    
    Returns:
        NDArray[np.float64]: Probability distribution where sum(output) = 1
    """
    # Scale inputs by temperature and compute exponentials
    exp_values = np.exp(arr / temperature)
    # Normalize by dividing by sum to get probability distribution
    return exp_values / np.sum(exp_values)

def plot_bar_chart(
    data: NDArray[np.float64],
    title: str,
    y_max: Optional[float] = None,
    color: str = 'skyblue'
) -> go.Figure:
    """
    Creates a bar chart visualization using Plotly.
    
    Args:
        data: Array of values to plot
        title: Chart title
        y_max: Optional maximum y-axis value
        color: Bar color (default: 'skyblue')
    
    Returns:
        go.Figure: Plotly figure object ready for display
    """
    # Create bar chart with indexed x-axis
    fig = go.Figure(data=[go.Bar(
        x=[str(i) for i in range(1, len(data)+1)],
        y=data,
        marker_color=color
    )])
    
    # Configure layout with proper labels and scaling
    fig.update_layout(
        title=title,
        xaxis_title='Index',
        yaxis_title='Value',
        yaxis=dict(
            # Auto-scale y-axis based on data range
            range=[
                min(data)*1.1 if min(data) < 0 else 0,
                y_max if y_max else max(data)*1.1
            ]
        ),
        template='plotly_white',
        height=400
    )
    return fig

def main():
    """
    Main application function that orchestrates the visualization pipeline.
    """
    # Get user inputs from sidebar
    input_numbers, temperature = create_sidebar_controls()
    
    # Calculate softmax distribution
    softmax_values = softmax(input_numbers, temperature)
    
    # Create and display visualizations
    # Original input values
    fig1 = plot_bar_chart(
        input_numbers,
        'Original Numbers (Input Values)',
        y_max=2
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    # Softmax transformed values
    fig2 = plot_bar_chart(
        softmax_values,
        'Softmax Values (Probability Distribution)',
        color='lightgreen'
    )
    st.plotly_chart(fig2, use_container_width=True)

if __name__ == "__main__":
    main()
