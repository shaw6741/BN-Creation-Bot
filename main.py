import os
import logging
import qrcode
import subprocess
import socket

def generate_qr_code(url, filename):
    qr = qrcode.QRCode()
    qr.add_data(url)
    qr.make(fit=True)
    qr_image = qr.make_image(fill_color="black", back_color="white")
    qr_image.save(filename)

def run_command(port):
    subprocess.Popen(['streamlit', 'run', './About.py', '--server.port', str(port)]) #change path here

def get_assigned_port():
    if os.path.isfile("port.txt"):
        with open("port.txt", "r") as file:
            port = int(file.read().strip())
    else:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', 0))
            _, port = s.getsockname()
        with open("port.txt", "w") as file:
            file.write(str(port))
    return port

def main():
    logging.info('Starting Engine')
    assigned_port = get_assigned_port()
    url = f"http://10.21.14.182:{assigned_port}"
    qr_code_filename = "./image_folder/qr_code.png" #change the path to save qr code here
    generate_qr_code(url, qr_code_filename)
    run_command(assigned_port)

if __name__ == '__main__':
    main()