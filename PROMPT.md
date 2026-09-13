# Karaokifex

Karaokifex is a Python command-line tool that will:

1. Take one mandatory parameter, a Youtube link, and optionally, an artist and a song name
2. It will use yt-dlp to download the best-quality video, including best quality audio stream
3. It will then separate the video from the audio stream (ffmpeg)
4. It will then pass the audio through audio_separator with `mel_band_roformer_karaoke_gabox.ckpt` to produce a karoke-ready version (song with no vocals but backing vocals) and a vocals-only version (use any model you think is best)
5. It will grab the best match of song lyrics from lrclib (artist + songname and best match in terms of song length)
6. It will also pass the vocals-only version through whisperx to get an exact timecoded log of which words
7. It will then combine the lrclib-lyrics with the whisperx timecodes to produce an ASS file that displays the lines as in the lyrics from lrclib, but with the highlights-progressing from whisperx, to produce a karaoke-style sub track
8. It will then combine the original video (make it slightly darker) with the song-with-backing-vocals-only audio and burn in the karaoke-style sub-track from the ASS step

## Implementation Details

* Use the already linked modules wherever possible
* Structure the project nicely so that a programmer can immediately understand it
* Use click for the command line tooling
* Use parallel processing whenever possible, do everything in parallel unless there's things that depend on other things
* Create beautiful command-line logging (but make it understandable with all the parallel processing taking place)
* For every individual song, create a subfolder in the current folder, where all temp-files and artifacts are stored - and add a prompt at the end before removing any temp files (and add a "autodelete" command line param to skip the prompt)
* Add unit tests as you see fit
