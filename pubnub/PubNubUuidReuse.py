from pubnub.pnconfiguration import PNConfiguration
from pubnub.pubnub import PubNub, SubscribeListener
import uuid
import random
import os


def get_random_uuid_as_string():
    basename_string = 'festeban26-iot-lockers-server'
    new_uuid_random_number = random.random()
    new_uuid_string = basename_string + "-" + str(new_uuid_random_number)
    custom_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, new_uuid_string)
    print("UUID generated: " + str(custom_uuid))
    return str(custom_uuid)


def get_uuid_from_default_uuid_file():
    # first check if a previous file exists, if it does not exist, generate uuid and save it to file
    try:
        with open("UUID.txt", "r") as text_file:
            return text_file.read()
    except IOError:
        print("UUID file was not found. Creating new uuid")
        new_uuid = get_random_uuid_as_string()
        save_uuid_in_default_uuid_file(new_uuid)
        with open("UUID.txt", "r") as text_file:
            return text_file.read()


def save_uuid_in_default_uuid_file(uuid_as_string):
    # first check if it exists, if it exists, delete the file
    try:
        file = open("UUID.txt", 'r')
        file.close()
        os.remove("UUID.txt")
    except IOError:
        pass
    # once the file is deleted or the file did not exists, save content
    with open("UUID.txt", "w") as text_file:
        text_file.write(uuid_as_string)


print(get_uuid_from_default_uuid_file())

'''
pub_num_iot_lockers_channel = 'iot-lockers'
pn_config = PNConfiguration()
pn_config.subscribe_key = 'INSERT API KEY'
pn_config.publish_key = 'INSERT API KEY'
pn_config.uuid = '4205e440-1258-462c-aa94-1870f1ab4755'
pub_nub = PubNub(pn_config)

print(pub_nub.uuid)
pub_nub.publish().channel(pub_num_iot_lockers_channel).message("Test").sync()
'''







