import time
import threading
from pathlib import Path
from collections import deque
import numpy as np
import soundfile as sf
import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox

from audio.recorder import Recorder
from audio.features import SAMPLE_RATE, DEFAULT_RECORD_SECONDS, DATA_DIR
from utils.types import RecordingResult
from model.trainer import train_model_from_folder, MODEL_PATH
from model.recognizer import recognize_file


class VoiceApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("dark-blue")
        self.title("Voice Recognition")
        self.geometry("780x500")
        self.resizable(False, False)

        self.recorder = Recorder()
        self.is_recording = False
        self.is_paused = False
        self.saving = False
        self.remaining_active = 0.0

        waveform_maxlen_seconds = 1.0
        self.waveform_maxlen_samples = int(SAMPLE_RATE * waveform_maxlen_seconds)
        self.wave_lock = threading.Lock()
        self.waveform_buffer = deque(maxlen=self.waveform_maxlen_samples)

        self._setup_ui()
        self.after(30, self._loop)

    def _setup_ui(self):
        # Left and right panels
        self.left = ctk.CTkFrame(self, corner_radius=12)
        self.left.place(relx=0.03, rely=0.03, relwidth=0.44, relheight=0.94)
        self.right = ctk.CTkFrame(self, corner_radius=12)
        self.right.place(relx=0.51, rely=0.03, relwidth=0.46, relheight=0.94)

        # Header
        ctk.CTkLabel(self.left, text="Classroom Voice Recognition",
                     font=ctk.CTkFont(size=18, weight="bold")).pack(pady=(12, 6))
        ctk.CTkLabel(self.left, text="Record student samples and train recognizer",
                     font=ctk.CTkFont(size=11), text_color="#a9b0b8").pack(pady=(0, 12))

        # Name + duration
        self.name_entry = ctk.CTkEntry(self.left, placeholder_text="Student Name (e.g. Ali)")
        self.name_entry.pack(padx=18, pady=(6, 12), fill="x")

        self.duration_combo = ctk.CTkOptionMenu(self.left,
                                                values=["2 sec", "3 sec", "4 sec", "5 sec", "6 sec", "8 sec"])
        self.duration_combo.set(f"{DEFAULT_RECORD_SECONDS} sec")
        self.duration_combo.pack(padx=18, pady=(0, 12))

        # Waveform canvas
        self.canvas_w, self.canvas_h = 320, 200
        self.canvas = tk.Canvas(self.left, width=self.canvas_w, height=self.canvas_h,
                                bg="#0f1113", highlightthickness=0)
        self.canvas.pack(padx=12, pady=(6, 6))
        self.countdown_label = ctk.CTkLabel(self.left, text="Ready", text_color="#cfe7ff")
        self.countdown_label.pack(padx=18, pady=(0, 8), anchor="c")

        # Record buttons
        self.button_frame = ctk.CTkFrame(self.left, fg_color="transparent")
        self.button_frame.pack(padx=18, pady=(6, 8), fill="x")

        self.record_btn = ctk.CTkButton(self.button_frame, text="🎙 Start Recording",
                                        fg_color="#1f6feb", hover_color="#1764c3",
                                        command=self.on_record)
        self.record_btn.pack(fill="x")

        # Controls during recording
        self.rec_ctrl_frame = ctk.CTkFrame(self.button_frame, fg_color="transparent")
        self.pause_btn = ctk.CTkButton(self.rec_ctrl_frame, text="⏸ Pause",
                                       fg_color="#c49b0b", hover_color="#a57f00",
                                       command=self.on_pause)
        self.discard_btn = ctk.CTkButton(self.rec_ctrl_frame, text="🗑 Discard",
                                         fg_color="#8b0000", hover_color="#a30000",
                                         command=self.on_discard)
        self.pause_btn.pack(side="left", expand=True, fill="x", padx=(0, 4))
        self.discard_btn.pack(side="left", expand=True, fill="x", padx=(4, 0))
        self.rec_ctrl_frame.pack_forget()

        # Train & Test buttons
        btn_pad = {"padx": 18, "pady": (6, 8), "fill": "x"}
        self.train_btn = ctk.CTkButton(self.left, text="⚙ Train Model", command=self.on_train)
        self.train_btn.pack(**btn_pad)
        self.test_btn = ctk.CTkButton(self.left, text="🔍 Test Recognition", command=self.on_test)
        self.test_btn.pack(**btn_pad)

        # ℹ️ Info label under Test button
        self.test_info_label = ctk.CTkLabel(
            self.left,
            text="(Voice recognition is available only after model training)",
            font=ctk.CTkFont(size=9),
            text_color="#9aa3ac"
        )
        self.test_info_label.pack(padx=18, pady=(0, 10))

        # ✅ Disable test button if model not trained
        self.test_btn.configure(state="normal" if MODEL_PATH.exists() else "disabled")

        # Status + log
        self.status_label = ctk.CTkLabel(self.left, text="Status: Ready",
                                         text_color="#cfd6dc", anchor="center")
        self.status_label.pack(padx=18, pady=(0, 12), fill="x")

        self.log_box = ctk.CTkTextbox(self.right, fg_color="#0e1114")
        self.log_box.pack(fill="both", expand=True, padx=12, pady=12)
        self.log("App ready.")

    # ---------------- RECORDING ----------------
    def on_record(self):
        if self.is_recording or self.saving:
            return
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showerror("Missing name", "Please enter student name.")
            return
        duration = int(self.duration_combo.get().split()[0])
        self._toggle_record_buttons(starting=True)
        threading.Thread(target=self._record_thread, args=(name, duration), daemon=True).start()

    def _toggle_record_buttons(self, starting: bool):
        if starting:
            self.record_btn.pack_forget()
            self.rec_ctrl_frame.pack(fill="x")
        else:
            self.rec_ctrl_frame.pack_forget()
            self.record_btn.pack(fill="x")
            self.is_paused = False
            self.saving = False
            self.countdown_label.configure(text="Ready")
            with self.wave_lock:
                self.waveform_buffer.clear()

    def on_pause(self):
        if not self.is_recording or self.saving:
            return
        self.is_paused = not self.is_paused
        self.recorder.pause(self.is_paused)
        if self.is_paused:
            self.pause_btn.configure(text="▶ Resume", fg_color="#3b8c2a")
            self.log("Recording paused.")
        else:
            self.pause_btn.configure(text="⏸ Pause", fg_color="#c49b0b")
            self.log("Recording resumed.")

    def on_discard(self):
        if self.is_recording:
            self._user_requested_discard = True
            self.log("Discard requested.")

    def _record_thread(self, name: str, duration: float):
        self.is_recording = True
        self.is_paused = False
        self._user_requested_discard = False
        self.remaining_active = duration
        self.train_btn.configure(state="disabled")
        self.test_btn.configure(state="disabled")
        self.set_status(f"Recording {name} for {duration}s")

        try:
            self.recorder.start()
        except Exception as e:
            messagebox.showerror("Recorder Error", str(e))
            self.is_recording = False
            self._toggle_record_buttons(False)
            self.train_btn.configure(state="normal")
            self.test_btn.configure(state="normal")
            return

        collected = []
        active_elapsed = 0.0
        prev_time = time.time()

        while (active_elapsed < duration) and (not self._user_requested_discard):
            now = time.time()
            dt = now - prev_time
            prev_time = now

            if not self.is_paused:
                active_elapsed += dt
                remaining = duration - active_elapsed
                self.remaining_active = max(0.0, remaining)
                frames = self.recorder.read_all()
                if frames.size:
                    collected.append(frames)
                    mono = frames.mean(axis=1) if frames.ndim > 1 else frames
                    with self.wave_lock:
                        self.waveform_buffer.extend(mono)
            time.sleep(0.016)

        self.remaining_active = 0.0
        leftover = self.recorder.read_all()
        if leftover.size:
            collected.append(leftover)
        self.recorder.stop()

        if collected:
            audio = np.vstack(collected).flatten()
        else:
            audio = None

        result = RecordingResult(name=name, audio=audio, discarded=self._user_requested_discard)
        self.after(0, lambda: self._handle_recording_result(result))

    def _handle_recording_result(self, result: RecordingResult):
        self.is_recording = False
        self._toggle_record_buttons(starting=False)
        self.train_btn.configure(state="normal")
        self.test_btn.configure(state="normal" if MODEL_PATH.exists() else "disabled")

        if result.discarded:
            self.log("Recording discarded by user.")
            return
        if result.audio is None or result.audio.size == 0:
            self.log("No audio captured.")
            return

        self.saving = True
        self.log("Saving recording...")

        def saver():
            folder = DATA_DIR / result.name
            folder.mkdir(parents=True, exist_ok=True)
            idx = len(list(folder.glob("*.wav"))) + 1
            path = folder / f"sample_{idx}.wav"
            try:
                sf.write(str(path), result.audio, SAMPLE_RATE)
                self.set_status(f"Saved: {path.name}")
            except Exception as e:
                self.log(f"Save failed: {e}")
            finally:
                self.saving = False
                with self.wave_lock:
                    self.waveform_buffer.clear()

        threading.Thread(target=saver, daemon=True).start()

    # ---------------- TRAIN / TEST ----------------
    def on_train(self):
        def _train():
            try:
                self.set_status("Training model...")
                acc = train_model_from_folder(status_callback=self.log)
                self.set_status(f"Trained ({acc*100:.1f}%)")
                messagebox.showinfo("Training done", f"Accuracy: {acc*100:.1f}%")
                self.test_btn.configure(state="normal")
            except Exception as e:
                messagebox.showerror("Training Error", str(e))
                self.set_status("Training failed.")
        threading.Thread(target=_train, daemon=True).start()

    def on_test(self):
        if self.is_recording or self.saving or not MODEL_PATH.exists():
            return
        duration = int(self.duration_combo.get().split()[0])
        threading.Thread(target=self._test_thread, args=(duration,), daemon=True).start()

    def _test_thread(self, duration: float):
        self.set_status("Recording test sample...")
        try:
            self.recorder.start()
        except Exception as e:
            self.log(f"Recorder start failed: {e}")
            return

        collected = []
        t0 = time.time()
        while time.time() - t0 < duration:
            frames = self.recorder.read_all()
            if frames.size:
                collected.append(frames)
                mono = frames.mean(axis=1) if frames.ndim > 1 else frames
                with self.wave_lock:
                    self.waveform_buffer.extend(mono)
            time.sleep(0.016)

        self.recorder.stop()
        if not collected:
            self.log("No audio captured for test.")
            return

        audio = np.vstack(collected).flatten()
        tmp = Path("temp_test.wav")
        try:
            sf.write(str(tmp), audio, SAMPLE_RATE)
            self.set_status("Recognizing...")
            pred, prob = recognize_file(tmp)
            messagebox.showinfo("Recognition", f"Predicted: {pred}\nConfidence: {prob*100:.1f}%")
            self.set_status(f"Predicted {pred} ({prob*100:.1f}%)")
        except Exception as e:
            self.log(f"Test error: {e}")
            self.set_status("Test failed.")
        finally:
            tmp.unlink(missing_ok=True)
            with self.wave_lock:
                self.waveform_buffer.clear()

    # ---------------- UTILITIES ----------------
    def log(self, text: str):
        stamp = time.strftime("%H:%M:%S")
        self.log_box.configure(state="normal")
        self.log_box.insert("end", f"[{stamp}] {text}\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def set_status(self, text: str):
        self.status_label.configure(text=f"Status: {text}")
        self.log(text)

    def _loop(self):
        self._draw_waveform()
        if self.is_recording and not self.is_paused and not self.saving:
            remaining = round(max(0.0, self.remaining_active), 1)
            self.countdown_label.configure(text=f"Remaining: {remaining:.1f}s")
        elif self.saving:
            dots = int((time.time() * 2) % 4)
            self.countdown_label.configure(text="Saving" + "." * dots)
        elif self.is_paused:
            self.countdown_label.configure(text=f"Paused — {self.remaining_active:.1f}s left")
        else:
            self.countdown_label.configure(text="Ready")
        self.after(30, self._loop)

    def _draw_waveform(self):
        self.canvas.delete("all")
        w, h = self.canvas_w, self.canvas_h
        mid = h // 2
        with self.wave_lock:
            buf = np.array(self.waveform_buffer, dtype=np.float32)
        if buf.size == 0:
            self.canvas.create_line(0, mid, w, mid, fill="#2e3440", width=2)
            return
        maxa = np.max(np.abs(buf)) or 1.0
        norm = buf / maxa
        if norm.size < w:
            pad = np.zeros(w - norm.size, dtype=np.float32)
            arr = np.concatenate([pad, norm])
        else:
            idxs = np.linspace(0, norm.size - 1, w).astype(int)
            arr = norm[idxs]
        self.canvas.create_line(0, mid, w, mid, fill="#172027", width=1)
        for i in range(1, w):
            x0, y0 = i - 1, mid - int(arr[i - 1] * (h // 2 - 6))
            x1, y1 = i, mid - int(arr[i] * (h // 2 - 6))
            self.canvas.create_line(x0, y0, x1, y1, fill="#66d9ff", width=2)


if __name__ == "__main__":
    app = VoiceApp()
    app.mainloop()
