from pubnub.pnconfiguration import PNConfiguration
from pubnub.pubnub import PubNub, SubscribeListener
from threading import Thread

pub_num_lockers_channel = 'iot-lockers'
lockers_owners_dic = {'Meli': 1, 'Pau': 2, 'Mare': 3, 'Noe': 4, 'Majo': 5, 'Nicky': 6, 'Jhes': 7, 'Mika': 8,
                      'Dome': 9, 'Esteban': 10, 'Pala': 11}
lockers_lock_status_is_open = {1: True, 2: True, 3: True, 4: True, 5: True, 6: True, 7: True, 8: True,
                               9: True, 10: True, 11: True}


def start_pub_nub_server(channel_name):
    pn_config = PNConfiguration()
    pn_config.subscribe_key = "INSERT API KEY"
    pn_config.publish_key = "INSERT API KEY"
    pn_config.uuid = '3242a702-6d6f-42d4-a548-356af0b95681'
    # instantiate a PubNub instance
    pub_nub = PubNub(pn_config)
    try:
        my_listener = SubscribeListener()
        pub_nub.add_listener(my_listener)
        pub_nub.subscribe().channels(channel_name).execute()
        my_listener.wait_for_connect()
        while True:
            result = my_listener.wait_for_message_on(channel_name)
            if result.message[1] == 'Server':  # if it is for me
                print("Just received a message: ", end='')
                print(result.message)
                petitioner = result.message[0]
                if petitioner in lockers_owners_dic:
                    if result.message[2] == 'locker_status':
                        locker_status = lockers_lock_status_is_open[lockers_owners_dic[petitioner]]
                        response = ['Server', petitioner]
                        if locker_status:
                            response.append("open")
                        else:
                            response.append("closed")
                        response.append(str(pn_config.uuid))
                        print("Responding: ", end='')
                        print(response)
                        pub_nub.publish().channel(pub_num_lockers_channel).message(response).sync()
                    elif result.message[2] == 'open':
                        # Call function to open the locker
                        print("Opening " + petitioner + "'s locker")
                        lockers_lock_status_is_open[lockers_owners_dic[petitioner]] = True
                        print(lockers_lock_status_is_open)
                    elif result.message[2] == 'close':
                        # Call function to close the locker
                        print("Closing " + petitioner + "'s locker")
                        lockers_lock_status_is_open[lockers_owners_dic[petitioner]] = False
                        print(lockers_lock_status_is_open)

    finally:
        pub_nub.unsubscribe().channels(channel_name).execute()
        pub_nub.stop()
        print("Unsubscribe")
    return


th = Thread(target=start_pub_nub_server, args=(pub_num_lockers_channel,))
th.start()
