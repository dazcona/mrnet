import telebot
import os

telegram_token = os.environ.get('telegram_token')
telegram_chat_id = os.environ.get('telegram_token')

def send(message):
	
	bot = telebot.TeleBot(telegram_token)
	bot.config['api_key'] = telegram_token
	bot.send_message(int(telegram_chat_id), message)
	print('[INFO] Sending text...')


if __name__ == "__main__":
	
	message = 'Hi! This is a test!'
	print('Sending text...{}'.format(message))
	send(message)