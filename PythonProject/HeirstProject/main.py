import turtle as t
import random

color_list = [(247, 240, 230), (234, 240, 246), (237, 247, 243), (248, 237, 242), (148, 165, 182)]

t.colormode(255)
tim = t.Turtle()
t.speed("fastest")
t.penup()
t.setheading(225)
t.forward(300)
t.setheading(0)

number_of_dots = 100

for dot_count in range(1,number_of_dots):
    t.dot(20,random.choice(color_list))
    t.forward(50)

    if dot_count % 10 == 0:
        t.setheading(90)
        t.forward(50)
        t.setheading(180)
        t.forward(500)
        t.setheading(0)


screen = t.Screen()

screen.exitonclick()