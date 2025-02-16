import time
from turtle import Screen , Turtle
from snake import Snake
from food import Food
from scoreboard import Scoreboard
screen = Screen()
screen.setup(width = 600, height =600)
screen.bgcolor("black")
screen.title("My Snake Game")
screen.tracer(0)

snake = Snake()
food = Food()
scoreboard = Scoreboard()
screen.listen()
screen.onkey(snake.up,"w")
screen.onkey(snake.left,"a")
screen.onkey(snake.down,"s")
screen.onkey(snake.right,"d")


game_on = True
while game_on:
    screen.update()
    time.sleep(0.1)
    snake.move()
    if snake.head.distance(food) < 15:
        food.refresh()
        snake.extend()
        scoreboard.increase_score()



    if snake.head.xcor() > 280 or snake.head.xcor() < -280 or snake.head.ycor() >280 or snake.head.ycor() < -280:

        scoreboard.reset()
        snake.reset()

    for segement in snake.segments[1:]:

        if snake.head.distance(segement) < 10:

            scoreboard.update_scoreboard()
            scoreboard.reset()
            snake.reset()

            


screen.exitonclick()