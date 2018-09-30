
import sqlite3
import threading
import numpy as np
from customWMD import WMD
import speech_recognition as speech
from sklearn.externals.joblib import Parallel, delayed

class Analyze:
	
	def __init__(self):
		
		self.conn = sqlite.connect("Command.db")
		self.curr = self.conn.cursor()
		self.recognizer = speech.Recognizer()
		self.wmd = WMD()
		self.thread = threading.Condition()

	def _get_from_raspberry(self):
		pass

	def _convert_to_audio(self):
		pass
	
	def _convert_to_text(self, audio: speech.AudioFile) -> str:
		'''
		audio: speech_recognition.AudioFile

		returns: text output from google api
		'''
		return self.recognizer.recognize_google(audio)
	
	def _retrieve_commands(self) -> list:
		'''
		returns: all commands from database
		'''
		self.curr.execute("SELECT command from commands")
		commands = self.curr.fetchall()

		return commands

	def _find_best(self, user_command: str, n_jobs=10) -> (np.float64, str):
		'''
		user_command: str is the command that user gave as input
		n_jobs: int for parallelizing

		returns: tuple (wmd calculated, command in the database)
		'''
		out = Parallel(n_jobs=n_jobs)(delayed(lambda x: (self.wmd.wmd(user_command, x), x)) for command in self.commands)
		out.sort(key = lambda x: x[0])

		return out[0]
	
	def wait_and_check(self):
		'''
			function which wakes up when command is given to the user 
		'''
			self.thread.wait()
			

if __name__ == "__main__":
	pass

