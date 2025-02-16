
import pandas
import turtle



screen = turtle.Screen()
screen.title("My US state game")
image = "blank_states_img.gif"
screen.addshape(image)
guessed_states = []
data = pandas.read_csv("50_states.csv")
all_states = data.state.to_list()


turtle.shape(image)
while len(guessed_states) < 50:
    answer_state = screen.textinput(title = "Guess the states?",prompt="What's another state.").title()

    if answer_state in all_states:
            t = turtle.Turtle()
            t.hideturtle()
            t.penup()
            state_data = data[data.state == answer_state]
            t.goto(state_data.x.item(),state_data.y.item())
            t.write(answer_state)
            guessed_states.append(answer_state)





screen.exitonclick()