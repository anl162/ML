# AI-Generated Image Detector

A classifier that distinguishes real photographs from AI-generated images, served through a web app where you upload an image and get a prediction.

Built at SPIS (Summer Program for Incoming Students), UC San Diego, Summer 2025.

## Result

70% accuracy on a held-out 15% test split of roughly 80,000 images (about 12,000 test images).

That number is modest and worth reading in context. The dataset pairs each real Shutterstock photo with an AI-generated counterpart of the same subject, so the model cannot lean on differences in content, lighting, or composition. It has to find the generator's fingerprint in the pixels themselves, and a fine-tuned ImageNet backbone at 224x224 resolution is a blunt instrument for that. Published detectors do considerably better, generally with far more data and with frequency-domain features that survive resizing and JPEG compression.

## Approach

I tried several framings before settling on one.

**What shipped: transfer learning with ResNet50 (TensorFlow / Keras).** ImageNet-pretrained ResNet50 with the classifier head removed, followed by global average pooling, a 128-unit dense layer with L2 regularization, batch normalization, dropout, and a sigmoid output. Training runs in two stages: first the backbone is frozen and only the new head trains (Adam at 3e-4), then the last 10 layers of the backbone are unfrozen and fine-tuned at 1e-5. Binary cross-entropy with label smoothing of 0.1, balanced class weights, early stopping on validation loss, and `ReduceLROnPlateau`. Augmentation is aggressive: random flips, rotation, zoom, contrast, and brightness.

**What I moved away from.** Earlier attempts trained with lighter augmentation and a larger unfrozen block (the last 20 layers, in `train(local).py`), which fit the training set quickly and generalized worse. Tightening regularization and unfreezing less is what got validation loss to stop diverging from training loss.

The honest summary: the interesting work here was diagnosing overfitting and responding to it, not the final number. A frozen ImageNet backbone is optimized for semantic content, and AI-vs-real is not a semantic distinction, which is most of why 70% is the ceiling this setup reaches.

## Dataset

[ShutterStock Dataset for AI vs Human-Gen. Image](https://www.kaggle.com/datasets/shreyasraghav/shutterstock-dataset-for-ai-vs-human-gen-image) on Kaggle, a mirror of the data from the [Detect AI vs. Human-Generated Images](https://www.kaggle.com/competitions/detect-ai-vs-human-generated-images) competition. Roughly 80,000 images, balanced between classes: real photographs from Shutterstock, each paired with an AI-generated image of the same subject. `train.csv` supplies `file_name` and a binary `label`.

I did not collect the images myself. `train_online.py` pulls the dataset at runtime through `kagglehub`, so the training script is reproducible without a manual download.

Splits are stratified with `random_state=42`: 70% train, 15% validation, 15% test.

## Repository layout

| File | What it is |
| --- | --- |
| `train_online.py` | Current training script. Downloads the dataset via `kagglehub` and runs the two-stage fine-tune. |
| `train(local).py` | Earlier version that reads the dataset from a local path. Kept for reference. |
| `training(outdated).ipynb` | The notebook the shipped `.h5` model was actually trained in (224x224 input). |
| `ai_detector.py` | Loads `ai_detector_model.h5` and returns a prediction plus confidence for one image. |
| `website/app.py` | Flask app: upload form, prediction route, result page. |
| `test_data/` | A handful of sample real and AI images for a quick sanity check. |

## Running it

```bash
pip install -r requirements.txt
python website/app.py
```

Then open http://localhost:3000 and upload an image.

To check the model straight from the command line without the web app:

```bash
python ai_detector.py
# Enter image path: test_data/ai/ai1.jpeg
```

To retrain (needs Kaggle API credentials configured for `kagglehub`):

```bash
python train_online.py
```

## Known issues

These are real and I would rather list them than let someone hit them cold.

- **Inference preprocessing does not match training.** `ai_detector.py` scales pixels with `/255.0`, but the shipped model was trained with `resnet50.preprocess_input`, which uses a different normalization. Served predictions are therefore worse than the reported test accuracy. This is the first thing to fix.
- **Input size drift.** The `.h5` model expects 224x224 (from the notebook), while `train_online.py` and `train(local).py` train at 160x160 and save to `.keras`. Retraining with the current scripts will not produce a drop-in replacement for the existing `.h5`.
- **Uploaded images do not render on the result page.** `app.py` saves uploads to `website/uploads/` but builds the preview URL as `static/uploads/...`, so the image 404s.
- **Class inference from text.** The `/predict` route decides the label by string-matching the word "ai" in the prediction text. It works, but it is fragile.

## Limitations

- Trained on a single dataset built from one generator pipeline. Accuracy against models it has not seen is untested and likely lower.
- At 70% accuracy, roughly one prediction in three is wrong. This is a learning project, not a usable detector.
- Images are resized before classification, which destroys much of the high-frequency signal that stronger detectors rely on.
- No calibration was done, so the confidence score the app displays should not be read as a probability.

## Stack

Python, TensorFlow / Keras, scikit-learn, NumPy, pandas, Matplotlib, Flask, kagglehub.
