# audio/recorder.py
import queue
import sounddevice as sd
import numpy as np

class Recorder:
    def __init__(self, samplerate=16000, channels=1, blocksize=1024):
        self.sr = samplerate
        self.channels = channels
        self.blocksize = blocksize
        self.q = queue.Queue()
        self.stream = None
        self.running = False
        self.paused = False

    def _callback(self, indata, frames, time_, status):
        if status:
            pass
        if self.running and not self.paused:
            self.q.put(indata.copy())

    def start(self):
        if self.running:
            return
        self.q = queue.Queue()
        self.stream = sd.InputStream(
            samplerate=self.sr, channels=self.channels,
            blocksize=self.blocksize, callback=self._callback
        )
        self.stream.start()
        self.running, self.paused = True, False

    def stop(self):
        if not self.running:
            return
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass
        self.running = False

    def pause(self, state: bool):
        self.paused = state

    def read_all(self):
        frames = []
        while True:
            try:
                frames.append(self.q.get_nowait())
            except queue.Empty:
                break
        if not frames:
            return np.empty((0, self.channels), dtype=np.float32)
        return np.vstack(frames)
