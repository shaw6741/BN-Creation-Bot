import os
import logging

def run_command():
    """
    Function to run the Streamlit application 'About.py'.

    This function uses the 'os.system' method to execute the 'streamlit run About.py' command,
    which runs the Streamlit application.

    Parameters:
        None

    Returns:
        None
    """
    os.system('streamlit run About.py')


def main():
    """
    Main function to start the engine and run the Streamlit application.

    This function logs the message 'Starting Engine' using the 'logging.info' method and then
    calls the 'run_command' function to execute the Streamlit application.

    Parameters:
        None

    Returns:
        None
    """
    logging.info('Starting Engine')
    run_command()

if __name__ == '__main__':
    main()