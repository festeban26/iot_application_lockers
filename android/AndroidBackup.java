package com.example.feste.pubnubhelloworld;

import android.os.Handler;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.CompoundButton;
import android.widget.TextView;
import android.widget.ToggleButton;
import android.widget.VideoView;

import com.pubnub.api.PNConfiguration;
import com.pubnub.api.PubNub;
import com.pubnub.api.callbacks.PNCallback;
import com.pubnub.api.callbacks.SubscribeCallback;
import com.pubnub.api.models.consumer.PNPublishResult;
import com.pubnub.api.models.consumer.PNStatus;
import com.pubnub.api.models.consumer.pubsub.PNMessageResult;
import com.pubnub.api.models.consumer.pubsub.PNPresenceEventResult;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


public class MainActivity extends AppCompatActivity {

    ToggleButton toggleButton;
    TextView statusTextView;
    VideoView videoView;
    String pubNubMyId = "Esteban";
    String pubNubChannel = "iot-lockers";
    PubNub pubnub;
    PNConfiguration pnConfiguration;
    Boolean isSettingStatus = false;

    private void updateStatus_toOpen() {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                statusTextView.setText("Estado: Abierto");
                toggleButton.setVisibility(View.VISIBLE);
                toggleButton.setEnabled(true);
                isSettingStatus = true;
                toggleButton.setChecked(false);
            }
        });
    }

    private void updateStatus_toClosed() {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                statusTextView.setText("Estado: Cerrado");
                toggleButton.setVisibility(View.VISIBLE);
                toggleButton.setEnabled(true);
                isSettingStatus = true;
                toggleButton.setChecked(true);
            }
        });
    }

    private void setStatus_toOpen() {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                toggleButton.setEnabled(false);
                if (isSettingStatus == false) {
                    statusTextView.setText("Abriendo");
                    // Enviar mensaje
                    pubnub.publish()
                            .message(Arrays.asList(pubNubMyId, "Server", "open"))
                            .channel(pubNubChannel)
                            .async(new PNCallback<PNPublishResult>() {
                                @Override
                                public void onResponse(PNPublishResult result, PNStatus status) {
                                    // handle publish result, status always present, result if successful
                                    // status.isError to see if error happened
                                }
                            });
                } else
                    isSettingStatus = false;

                new Handler().postDelayed(new Runnable() {

                    @Override
                    public void run() {
                        // This method will be executed once the timer is over
                        updateStatus_toOpen();
                        isSettingStatus = false;

                    }
                }, 1000);// set time as per your requirement
            }
        });
    }

    private void setStatus_toClosed() {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                toggleButton.setEnabled(false);
                if (isSettingStatus == false) {
                    statusTextView.setText("Cerrando");
                    // Enviar mensaje
                    pubnub.publish()
                            .message(Arrays.asList(pubNubMyId, "Server", "close"))
                            .channel(pubNubChannel)
                            .async(new PNCallback<PNPublishResult>() {
                                @Override
                                public void onResponse(PNPublishResult result, PNStatus status) {
                                    // handle publish result, status always present, result if successful
                                    // status.isError to see if error happened
                                }
                            });
                } else
                    isSettingStatus = false;


                new Handler().postDelayed(new Runnable() {

                    @Override
                    public void run() {
                        // This method will be executed once the timer is over
                        updateStatus_toClosed();
                        isSettingStatus = false;

                    }
                }, 1000);// set time as per your requirement
            }
        });
    }


    @Override
    protected void onResume() {
        super.onResume();
        isSettingStatus = false;
        toggleButton = findViewById(R.id.toggleButton);
        statusTextView = findViewById(R.id.textView_Status);
        toggleButton.setVisibility(View.INVISIBLE);
        toggleButton.setEnabled(false);
        statusTextView.setText("Cargando...");

        // As son as the app opens, request my locker status
        pnConfiguration = new PNConfiguration();
        pnConfiguration.setSubscribeKey("sub-c-cfc2d9cc-66c1-11e8-9499-26ede0a09d22");
        pnConfiguration.setPublishKey("pub-c-16b53bd0-0d96-4e65-9e77-b96f3a8b18b7");
        pubnub = new PubNub(pnConfiguration);
        pubnub.addListener(new SubscribeCallback() {
            @Override
            public void status(PubNub pubNub, PNStatus pnStatus) {

            }

            @Override
            public void message(PubNub pubNub, PNMessageResult pnMessageResult) {
                String messageAsString = pnMessageResult.getMessage().toString();
                messageAsString = messageAsString.replace('[', ' ')
                        .replace(']', ' ')
                        .replace('"', ' ');
                List<String> list = new ArrayList<String>(Arrays.asList(messageAsString.trim().split(" , ")));
                if (list.get(1).equals(pubNubMyId)) {
                    if (list.get(2).equals("open")) {
                        updateStatus_toOpen();
                    } else if (list.get(2).equals("closed")) {
                        updateStatus_toClosed();
                    }
                }
            }

            @Override
            public void presence(PubNub pubNub, PNPresenceEventResult pnPresenceEventResult) {

            }
        });

        pubnub.subscribe().channels(Arrays.asList(pubNubChannel)).execute();

        pubnub.publish()
                .message(Arrays.asList(pubNubMyId, "Server", "locker_status"))
                .channel(pubNubChannel)
                .async(new PNCallback<PNPublishResult>() {
                    @Override
                    public void onResponse(PNPublishResult result, PNStatus status) {
                        // handle publish result, status always present, result if successful
                        // status.isError to see if error happened
                    }
                });
    }

    @Override
    protected void onPause() {
        super.onPause();
        pubnub.unsubscribe();
        pubnub.destroy();
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        toggleButton = findViewById(R.id.toggleButton);

        videoView = findViewById(R.id.videoView);
        videoView.setVisibility(View.INVISIBLE);

        toggleButton.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                if (isChecked) {
                    setStatus_toClosed();

                } else {
                    setStatus_toOpen();
                    Log.d("E", "toOpen");
                    //statusTextView.setText("is not Checked");
                    /*
                    Uri uri = Uri.parse("android.resource://"+getPackageName() + "/" + R.raw.padlock_animation);
                    videoView.setVideoURI(uri);
                    videoView.setVisibility(View.VISIBLE);
                    videoView.setBackgroundColor(Color.WHITE);
                    videoView.start();
                    videoView.setOnPreparedListener(new MediaPlayer.OnPreparedListener() {
                        @Override
                        public void onPrepared(MediaPlayer mp) {
                            videoView.setBackgroundColor(Color.TRANSPARENT);
                        }
                    });*/
                }
            }
        });
    }
}
