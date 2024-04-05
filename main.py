import cv2
import discord
import time
from discord.ext import commands

with open('token.txt', 'r') as file:
    token = file.read().strip()

with open('channel.txt', 'r') as file:
    channel_num = int(file.read().strip())

bot = commands.Bot(command_prefix="!", intents=discord.Intents.all())

class MyCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.last_detection_time = 0

    @bot.event
    async def send_message(self):
        current_time = time.time()
        # Check if ten second has passed since the last detection
        if current_time - self.last_detection_time >= 3:
            channel = bot.get_channel(channel_num)
            if channel:
                await channel.send("Motion Detected")
            self.last_detection_time = current_time

@bot.event
async def on_ready():
    print('Bot is ready!')
    await bot.add_cog(MyCog(bot))
    await detect_motion()

async def detect_motion():
    # Initialize camera
    camera = cv2.VideoCapture(0)

    # Capture first frame
    _, prev_frame = camera.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    while True:
        # Capture frame
        _, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate frame difference
        frame_diff = cv2.absdiff(prev_gray, gray)
        
        # Apply threshold
        _, thresh = cv2.threshold(frame_diff, 5, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Use this if you don't want to display what the camera is recording
        # # Check for motion
        # motion_detected = any(cv2.contourArea(contour) > 500 for contour in contours)
        #
        # # If motion is detected, send a message
        # if motion_detected:
        #     await bot.get_cog('MyCog').send_message()
        
        # Use this if you do want to display what the camera is recording
        # Draw bounding boxes around detected movement
        for contour in contours:
            if cv2.contourArea(contour) > 1000:  # Adjust this threshold as needed
                await bot.get_cog('MyCog').send_message()
        
        # Display frame
        cv2.imshow('Motion Detection', frame)

        # Update previous frame
        prev_gray = gray.copy()
        
        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release camera and close windows
    camera.release()
    cv2.destroyAllWindows()

bot.run(token)
