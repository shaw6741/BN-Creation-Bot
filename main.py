import os
import logging

def run_command():
    os.system('streamlit run E:/px/UChi/Courses/Capstone/code/About.py')


def main():
    logging.info('Starting Engine')
    run_command()


if __name__ == '__main__':
    main()