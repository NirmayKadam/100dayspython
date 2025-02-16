from turtle import Turtle,Screen
import random
timmy = Turtle()
colours = ["CornflowerBlue", "DarkOrchid", "IndianRed", "DeepSkyBlue", "LightSeaGreen", "wheat", "SlateGray", "SeaGreen"]


directions = [0,90,180,270]
timmy.pensize(10)
timmy.speed(10)

for _ in range(200):
    timmy.forward(50)
    timmy.right(random.choice(directions))
    timmy.color(random.choice(colours))



screen = Screen()
screen.exitonclick()