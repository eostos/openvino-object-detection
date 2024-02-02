import string
import threading
import base64
import json
import json
import os
import redis
import cv2
import traceback

#import easyocr

# Initialize the OCR reader
#reader = easyocr.Reader(['en'], gpu=False)

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}

def is_running_in_docker():
    path = "/.dockerenv"
    return os.path.exists(path)

def send_video(image, connect_redis,device_id):
    # Resize the image to 320x240
    resized_image = cv2.resize(image, (320, 240))
    # Crop the bottom half of the image
    H = resized_image.shape[0]
    displ = resized_image[H//2:H, :, :]
    # Convert image to JPEG format
    jpg_string = cv2.imencode('.jpg', resized_image)[1]
    # Encode the image in base64
    encoded_string = base64.b64encode(jpg_string)
    # Publish to Redis
    connect_redis.publish("lpr_streaming_"+device_id, encoded_string)

def write_csv(results, output_path):
    """
    Write the results to a CSV file.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score'))

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                print(results[frame_nmr][car_id])
                if 'car' in results[frame_nmr][car_id].keys() and \
                   'license_plate' in results[frame_nmr][car_id].keys() and \
                   'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                            car_id,
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['car']['bbox'][0],
                                                                results[frame_nmr][car_id]['car']['bbox'][1],
                                                                results[frame_nmr][car_id]['car']['bbox'][2],
                                                                results[frame_nmr][car_id]['car']['bbox'][3]),
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][0],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][1],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][2],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][3]),
                                                            results[frame_nmr][car_id]['license_plate']['bbox_score'],
                                                            results[frame_nmr][car_id]['license_plate']['text'],
                                                            results[frame_nmr][car_id]['license_plate']['text_score'])
                            )
        f.close()


def license_complies_format(text):
    """
    Check if the license plate text complies with the required format.

    Args:
        text (str): License plate text.

    Returns:
        bool: True if the license plate complies with the format, False otherwise.
    """
    if len(text) != 7:
        return False

    if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
       (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
       (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in dict_char_to_int.keys()) and \
       (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
       (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
       (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and \
       (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys()):
        return True
    else:
        return False


def format_license(text):
    """
    Format the license plate text by converting characters using the mapping dictionaries.

    Args:
        text (str): License plate text.

    Returns:
        str: Formatted license plate text.
    """
    license_plate_ = ''
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char, 5: dict_int_to_char, 6: dict_int_to_char,
               2: dict_char_to_int, 3: dict_char_to_int}
    for j in [0, 1, 2, 3, 4, 5, 6]:
        if text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]

    return license_plate_


# def read_license_plate(license_plate_crop):
#     """
#     Read the license plate text from the given cropped image.

#     Args:
#         license_plate_crop (PIL.Image.Image): Cropped image containing the license plate.

#     Returns:
#         tuple: Tuple containing the formatted license plate text and its confidence score.
#     """

#     detections = reader.readtext(license_plate_crop)

#     for detection in detections:
#         bbox, text, score = detection

#         text = text.upper().replace(' ', '')

#         if license_complies_format(text):
#             return format_license(text), score

#     return None, None


def get_car(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.

    Args:
        license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.

    Returns:
        tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
    """
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1
# Handler Params

################################################
################################################


def getConfigs(file_path, is_docker=False):
    try:
        # Check if the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file at {file_path} does not exist.")

        with open(file_path, 'r') as file:
            data = json.load(file)
            
            if is_docker:
                
                data = data[0]['params'][0]

            # Extracting fields
            manager_id = data['manager_id']
            device_id = data['device_id']
            ip_redis = data['ip_redis']
            port_redis = data['port_redis']
            ip_rest = data['ip_rest']
            vid_path = data['vid_path']
            fps = data['fps']
            ocr_port = data['ocr_port']
            alter_config_roi_scale = data['alter-config']['roi_scale']
            use_OCR = data['use_OCR']
            model = data['model']
            debug = data['debug']
            architecture_type=data['architecture_type']
            ocr_http = data['ocr_http']
            ocr_ip = data['ocr_ip']
            country = data['country']
            device = data['device']
            factor_width =data['factor_width']
            factor_height = data['factor_height']
            prom_frame= data["prom_frame"]
            ocr_http = data['ocr_http']
            ocr_grcp = data["ocr_grcp"]
            ocr_grcp_port = data["ocr_grcp_port"]
            ocr_grcp_ip = data["ocr_grcp_ip"]
            treshold_plate = data["treshold_plate"]
            regular_expressions = data["regular_expressions"]
            alter_config = data["alter-config"]
            limit_ram = data['limit_ram']
            



            extracted_fields = {
                'manager_id': manager_id,
                'device_id': device_id,
                'ip_redis': ip_redis,
                'port_redis': port_redis,
                'ip_rest': ip_rest,
                'vid_path': vid_path,
                'debug':debug,
                'fps': fps,
                'ocr_port': ocr_port,
                'ocr_http':ocr_http,
                'ocr_ip':ocr_ip,
                'alter_config_roi_scale': alter_config_roi_scale,
                'use_OCR': use_OCR,
                'model': model,
                'architecture_type':architecture_type,
                'country':country,
                'device':device,
                'factor_width':factor_width,
                'factor_height':factor_height,
                'prom_frame':prom_frame,
                'treshold_plate':treshold_plate,
                'ocr_grcp_ip':ocr_grcp_ip,
                'ocr_grcp_port':ocr_grcp_port,
                'ocr_grcp':ocr_grcp,
                'regular_expressions':regular_expressions,
                'ocr_http' : ocr_http,
                'alter_config':alter_config,
                'limit_ram':limit_ram

                
            }
            
            return json.dumps(extracted_fields, indent=4)

    except FileNotFoundError as e:
        print(f"File Error: {e}")
        return None
    except KeyError as e:
        traceback.print_exc()
        print(f"Error: Missing field in JSON data - {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON data, PLEASE VERIFY IT - {e}")
        return None


