# Sample Emotion Videos

This folder contains 7 short sample videos for testing the uploaded-video emotion detection flow:

- `angry.mp4`
- `disgust.mp4`
- `fear.mp4`
- `happy.mp4`
- `neutral.mp4`
- `sad.mp4`
- `surprise.mp4`

It also contains a recommended FER-style sample set in:

- `fer_style/angry.mp4`
- `fer_style/disgust.mp4`
- `fer_style/fear.mp4`
- `fer_style/happy.mp4`
- `fer_style/neutral.mp4`
- `fer_style/sad.mp4`
- `fer_style/surprise.mp4`

These `fer_style` videos were derived from the original clips by making them closer to FER-2013 style:

- tighter face crop
- grayscale frames
- histogram equalization
- fixed square output size

These do not change the true model accuracy, but they usually make demo predictions look more stable because the input is closer to the training distribution.

Source:

- RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
- Zenodo record: https://zenodo.org/records/1188976

These clips were extracted from the public `Video_Speech_Actor_01.zip` download and renamed for quick testing.

Emotion mapping used from the RAVDESS naming convention:

- `01 = neutral`
- `03 = happy`
- `04 = sad`
- `05 = angry`
- `06 = fearful`
- `07 = disgust`
- `08 = surprised`

License note:

- RAVDESS is released under `CC BY-NC-SA 4.0`
- If you reuse these sample files outside local testing, keep the dataset attribution and license terms
