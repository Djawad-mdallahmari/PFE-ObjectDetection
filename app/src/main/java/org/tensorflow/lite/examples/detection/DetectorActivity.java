/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.detection;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.Point;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.Build;
import android.os.Handler;
import android.os.SystemClock;
import android.os.VibrationEffect;
import android.os.Vibrator;
import android.speech.tts.TextToSpeech;
import android.util.Size;
import android.util.TypedValue;
import android.widget.Toast;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import org.tensorflow.lite.examples.detection.customview.OverlayView;
import org.tensorflow.lite.examples.detection.customview.OverlayView.DrawCallback;
import org.tensorflow.lite.examples.detection.customview.RecognitionScoreView;
import org.tensorflow.lite.examples.detection.env.BorderedText;
import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.tflite.Detector;
import org.tensorflow.lite.examples.detection.tflite.TFLiteObjectDetectionAPIModel;
import org.tensorflow.lite.examples.detection.tracking.MultiBoxTracker;

/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
public class DetectorActivity extends CameraActivity implements OnImageAvailableListener {
  private static final Logger LOGGER = new Logger();

  // Configuration values for the prepackaged SSD model.
  private static final int TF_OD_API_INPUT_SIZE = 300;
  private static final boolean TF_OD_API_IS_QUANTIZED = true;
  private static final String TF_OD_API_MODEL_FILE = "detect.tflite"; // Le modèle en format tflite
  private static final String TF_OD_API_LABELS_FILE = "labelmap.txt"; // Les labels accompagnant le modèle
  private static final DetectorMode MODE = DetectorMode.TF_OD_API;
  // Minimum detection confidence to track a detection.
  private static float MINIMUM_CONFIDENCE_TF_OD_API = 0.5f; // Le niveau de precision minimum en % (entre 0 et 1)
  private static final boolean MAINTAIN_ASPECT = false;
  private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);
  private static final boolean SAVE_PREVIEW_BITMAP = false;
  private static final float TEXT_SIZE_DIP = 10;
  OverlayView trackingOverlay;
  private Integer sensorOrientation;

  private Detector detector;

  private long lastProcessingTimeMs;
  private Bitmap rgbFrameBitmap = null;
  private Bitmap croppedBitmap = null;
  private Bitmap cropCopyBitmap = null;

  private boolean computingDetection = false;

  private long timestamp = 0;

  private Matrix frameToCropTransform;
  private Matrix cropToFrameTransform;

  private MultiBoxTracker tracker;

  private BorderedText borderedText;

  private TextToSpeech tts; // L'objet permettant de faire de la synthèse vocale
  private String readedText = ""; // Variable utilisé pour stoquer le dernier label de l'objet prononcé
  private RectF viseur; // Utilisé pour représenté le rectangle du viseur
  private Vibrator vibrator; // Pour la vibration

  @Override
  public void onPreviewSizeChosen(final Size size, final int rotation) {
    //On entre dans cette fonction qu'une fois au début pour initaliser la taille des input, le détecteur, l'orientation de la camera, ...
    final float textSizePx =
        TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);
    borderedText.setTypeface(Typeface.MONOSPACE);

    tracker = new MultiBoxTracker(this);

    int cropSize = TF_OD_API_INPUT_SIZE;

    try {
      detector =
          TFLiteObjectDetectionAPIModel.create(
              this,
              TF_OD_API_MODEL_FILE,
              TF_OD_API_LABELS_FILE,
              TF_OD_API_INPUT_SIZE,
              TF_OD_API_IS_QUANTIZED);
      cropSize = TF_OD_API_INPUT_SIZE;
    } catch (final IOException e) {
      e.printStackTrace();
      LOGGER.e(e, "Exception initializing Detector!");
      Toast toast =
          Toast.makeText(
              getApplicationContext(), "Detector could not be initialized", Toast.LENGTH_SHORT);
      toast.show();
      finish();
    }

    previewWidth = size.getWidth();
    previewHeight = size.getHeight();

    sensorOrientation = rotation - getScreenOrientation();
    LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

    LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
    rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
    croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888);

    frameToCropTransform =
        ImageUtils.getTransformationMatrix(
            previewWidth, previewHeight,
            cropSize, cropSize,
            sensorOrientation, MAINTAIN_ASPECT);

    cropToFrameTransform = new Matrix();
    frameToCropTransform.invert(cropToFrameTransform);

    trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
    trackingOverlay.addCallback(
        new DrawCallback() {
          @Override
          public void drawCallback(final Canvas canvas) {
            tracker.draw(canvas);
            if (isDebug()) {
              tracker.drawDebug(canvas);
            }
          }
        });

    tracker.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation);
    vibrator = (Vibrator) getSystemService(Context.VIBRATOR_SERVICE);
  }

  @Override
  protected void processImage() { // On entre dans cette fonction dès qu'une image (frame) est disponible <=> très fréquemment
    //Put elsewehre (TODO : à mettre au dessus pour être initialisé qu'une fois)
    if(tts == null){
      tts = new TextToSpeech(this, new TextToSpeech.OnInitListener() {
        @Override
        public void onInit(int status) {

        }
      });
      tts.setLanguage(Locale.US); // TODO : put in french. (But model's metadata (labels) are in english !)
    }

    //Ci-dessous les tentatives que j'ai faites obtenir le centre de l'écran... à revoir

    //System.out.println(left+" "+top+" "+right+" "+bottom);
    //viseur = new RectF(left,top,right,bottom);
    viseur = new RectF(140,180,160,200);
    /*int mWidth= this.getResources().getDisplayMetrics().widthPixels;
    int mHeight= this.getResources().getDisplayMetrics().heightPixels;
    viseur = new RectF(mWidth/2-100,mHeight/4-100,mWidth/2+100,mHeight/4+100);*/
    /*System.out.println("["+trackingOverlay.getLeft()+"] ["+trackingOverlay.getTop()+"] ["+trackingOverlay.getRight()+"] ["+trackingOverlay.getBottom()+"]");
    viseur = new RectF(trackingOverlay.getLeft()+(trackingOverlay.getRight()-trackingOverlay.getLeft())/3, //360
            trackingOverlay.getTop()+(trackingOverlay.getBottom()-trackingOverlay.getTop())/3, //673
            trackingOverlay.getRight()-(trackingOverlay.getRight()-trackingOverlay.getLeft())/3, //720
            trackingOverlay.getBottom()-(trackingOverlay.getBottom()-trackingOverlay.getTop())/3); //1346*/



    ++timestamp;
    final long currTimestamp = timestamp;
    trackingOverlay.postInvalidate();

    // No mutex needed as this method is not reentrant.
    if (computingDetection) {
      readyForNextImage();
      return;
    }
    computingDetection = true;
    LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

    rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

    readyForNextImage();

    final Canvas canvas = new Canvas(croppedBitmap);
    canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
    // For examining the actual TF input.
    if (SAVE_PREVIEW_BITMAP) {
      ImageUtils.saveBitmap(croppedBitmap);
    }

    runInBackground(
        new Runnable() {
          private boolean isWaiting = false;

          @Override
          public void run() {
            LOGGER.i("Running detection on image " + currTimestamp);
            final long startTime = SystemClock.uptimeMillis();
            final List<Detector.Recognition> results = detector.recognizeImage(croppedBitmap); //La reconnaissance est lancé ici, on recupere la liste des résultats (results)
            lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

            //Pour les boites englobantes
            cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
            final Canvas canvas = new Canvas(cropCopyBitmap);
            final Paint paint = new Paint();
            paint.setColor(Color.RED);
            paint.setStyle(Style.STROKE);
            paint.setStrokeWidth(2.0f);
            // Pour que le rectangle du viseur affiché, je l'ajoute dans la liste des résultats (pas top)
            Detector.Recognition viseurReco = new Detector.Recognition("viseur","viseur",1.0f,viseur);
            results.add(viseurReco);

            float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
            switch (MODE) {
              case TF_OD_API:
                minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                break;
            }

            final List<Detector.Recognition> mappedRecognitions =
                new ArrayList<Detector.Recognition>();

            for (final Detector.Recognition result : results) {
              final RectF location = result.getLocation();
              if (location != null && result.getConfidence() >= minimumConfidence) {
                canvas.drawRect(location, paint);

                //if(!result.equals(viseurReco))
                cropToFrameTransform.mapRect(location);

                result.setLocation(location);
                mappedRecognitions.add(result);
              }
            }

            tracker.trackResults(mappedRecognitions, currTimestamp);
            trackingOverlay.postInvalidate();

            computingDetection = false;

            float finalMinimumConfidence = minimumConfidence;
            runOnUiThread(
                new Runnable() {
                  @Override
                  public void run() {
                    showFrameInfo(previewWidth + "x" + previewHeight);
                    showCropInfo(cropCopyBitmap.getWidth() + "x" + cropCopyBitmap.getHeight());
                    showInference(lastProcessingTimeMs + "ms");

                    // Ici est la boucle de traitement des résultats
                    for (Detector.Recognition r :results) {
                      if(r.getConfidence() >= finalMinimumConfidence){
                        System.out.println(r.getTitle()+" "+r.getLocation().toShortString()+ " | " + viseur.toShortString() + " | "+r.getLocation().centerX()+" , "+r.getLocation().centerY());
                        if(!r.equals(viseurReco)) // Si c'est PAS le viseur (car viseur ajouté dans la liste des résultats pour qu'il soit affiché, voir plus haut)

                          if(r.getLocation().contains(viseurReco.getLocation())){ //Si le viseur est dans l'objet détecté && pas le même
                            if (readedText.equals(r.getTitle())){ // Si c'est le même
                              if(!tts.isSpeaking()){
                                vibrator.vibrate(200); // Vibre
                                tts.playSilentUtterance(3000,TextToSpeech.QUEUE_FLUSH,null);
                                readedText = "";
                              }
                            }else{
                              if(!tts.isSpeaking()){
                                tts.speak(r.getTitle(), TextToSpeech.QUEUE_FLUSH, null, null); // Synthetiseur prononce le label de l'objet
                                //vibrator.vibrate(200); // Vibre
                                readedText = r.getTitle();
                              }
                            }

                              /*final Handler handler = new Handler();
                              handler.postDelayed(new Runnable() {
                                @Override
                                public void run() {
                                  tts.speak(r.getTitle(), TextToSpeech.QUEUE_FLUSH, null, null); // Synthetiseur prononce le label de l'objet
                                  vibrator.vibrate(200); // Vibre
                                  readedText = r.getTitle();
                                }
                              },5000);*/
                            }

                          }
                    }


                  }
                });
          }
        });
  }

  @Override
  protected int getLayoutId() {
    return R.layout.tfe_od_camera_connection_fragment_tracking;
  }

  @Override
  protected Size getDesiredPreviewFrameSize() {
    return DESIRED_PREVIEW_SIZE;
  }

  // Which detection model to use: by default uses Tensorflow Object Detection API frozen
  // checkpoints.
  private enum DetectorMode {
    TF_OD_API;
  }

  @Override
  protected void setUseNNAPI(final boolean isChecked) {
    runInBackground(
        () -> {
          try {
            detector.setUseNNAPI(isChecked);
          } catch (UnsupportedOperationException e) {
            LOGGER.e(e, "Failed to set \"Use NNAPI\".");
            runOnUiThread(
                () -> {
                  Toast.makeText(this, e.getMessage(), Toast.LENGTH_LONG).show();
                });
          }
        });
  }

  @Override
  protected void setNumThreads(final int numThreads) {
    runInBackground(
        () -> {
          try {
            detector.setNumThreads(numThreads);
          } catch (IllegalArgumentException e) {
            LOGGER.e(e, "Failed to set multithreads.");
            runOnUiThread(
                () -> {
                  Toast.makeText(this, e.getMessage(), Toast.LENGTH_LONG).show();
                });
          }
        });
  }

  @Override
  protected void setNumSeuil(int numSeuil) {
    runInBackground(
            () -> {
              try {
                MINIMUM_CONFIDENCE_TF_OD_API = numSeuil/100f;
                runOnUiThread(
                        () -> {
                          Toast.makeText(this, "Seuil : "+numSeuil+"%", Toast.LENGTH_LONG).show();
                        });
              } catch (IllegalArgumentException e) {
                LOGGER.e(e, "Failed to set seuil.");
                runOnUiThread(
                        () -> {
                          Toast.makeText(this, e.getMessage(), Toast.LENGTH_LONG).show();
                        });
              }
            });
  }
}
