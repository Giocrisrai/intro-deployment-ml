import os
from base64 import b64encode

def main():
    key = os.environ.get('SERVICE_ACCOUNT_KEY')
    
    # Decodifica la clave y luego codifica con base64
    decoded_key = b64encode(key.encode()).decode()
    
    with open('path.json', 'w') as json_file:
        json_file.write(decoded_key)

    print(os.path.realpath('path.json'))

if __name__ == '__main__':
    main()
