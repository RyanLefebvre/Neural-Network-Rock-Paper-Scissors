from keras.models import load_model
import cv2
import numpy as np
from random import choice

# Inverted map of numeric values to classes.
REV_CLASS_MAP = {
    0: "rock",
    1: "paper",
    2: "scissors",
    3: "none"
}

# Returns the inverse of the getter
# from the train_Model.py file.
def getInvertedClassValue(val):
    return REV_CLASS_MAP[val]

# Returns the winner of a rock paper 
# scissors game between the user and CPU.
def calculate_winner(move1, move2):
    if move1 == move2:
        return "Tie"
    if move1 == "rock":
        if move2 == "scissors":
            return "User"
        if move2 == "paper":
            return "Computer"
    if move1 == "paper":
        if move2 == "rock":
            return "User"
        if move2 == "scissors":
            return "Computer"
    if move1 == "scissors":
        if move2 == "paper":
            return "User"
        if move2 == "rock":
            return "Computer"

# Loads the model and starts an infinite loop that 
# will calculate a game winner as long as the user
# input is not 'none'
model = load_model("rock-paper-scissors-model.h5")
cap = cv2.VideoCapture(0)
cap.set(3,960)
cap.set(4,540)
prev_move = None
while True:
    ret, frame = cap.read()
    if not ret:
        continue
    cv2.rectangle(frame, (75, 75), (325, 325), (255,192,203), 2)
    roi = frame[75:325, 75:325]
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (250, 250))
    pred = model.predict(np.array([img]))
    move_code = np.argmax(pred[0])
    user_move_name = getInvertedClassValue(move_code)
    if prev_move != user_move_name:
        if user_move_name != "none":
            computer_move_name = choice(['rock', 'paper', 'scissors'])
            winner = calculate_winner(user_move_name, computer_move_name)
        else:
            computer_move_name = "none"
            winner = "Waiting..."
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "User Move: " + user_move_name,
                (50, 50), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Computer Move: " + computer_move_name,
                (450, 50), font, 1.2, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Winner: " + winner,
                (70, 350), font, 0.8, (0, 0, 255), 4, cv2.LINE_AA)
    cv2.imshow("Rock Paper Scissors", frame)
    lastMoveWasNone = (prev_move == "none")
    currentMoveIsNotNone  = (user_move_name != "none")
    if(lastMoveWasNone and currentMoveIsNotNone):
        print("----BEGIN ROUND----")
        print("\tUser Move " + user_move_name )
        print("\tComputer Move " + computer_move_name)
        print("\t\tThe Winner is " + winner)
    prev_move = user_move_name
    keyPressed = cv2.waitKey(10)
    shouldKillProgram = (keyPressed == ord('q'))
    if(shouldKillProgram):
        break
cap.release()
cv2.destroyAllWindows()
