from manim import *
import numpy as np

anispeed = 0.5
changeColor = RED

class VideoAttempt(MovingCameraScene):
    def construct(self):
        circle = Circle()  # create a circle
        circles = []
        circles.append(circle)
        layer2 = []
        layer3 = []
        layer4 = []
        output = Circle()
        numCircs = 4
        numHiddens = 5
        for x in range(numCircs):
            circles.append(Circle())
        prevCirc = circle
        
        for circ in circles:
            circ.set_color(WHITE)
        
        makeLayer(self, layer2, numHiddens, 8)
        makeLayer(self, layer3, numHiddens, 0)
        makeLayer(self, layer4, numHiddens, -8)
        
        lines23 = connectLayers(layer2, layer3)
        lines34 = connectLayers(layer3, layer4)

        output.shift(LEFT*-17)
        output.set_color(WHITE)
        
        self.camera.frame.save_state()
        self.play(Create(circle), run_time = anispeed)  # show the square on screen
        
        self.play(self.camera.frame.animate.set(width = circle.width*30), run_time = anispeed)
        self.play(circle.animate.shift(DOWN*5 + LEFT*17), run_time = anispeed)

        prevCirc = circle
        for i, circ in enumerate(circles): 
            if(circ != prevCirc):
                circ.next_to(prevCirc, UP, buff = 0.5)
            prevCirc = circ

        for circ in circles:
            if circ != circle:
                circ.set_color(changeColor)
                self.play(Create(circ), run_time = anispeed)
                self.play(circ.animate.set_color(WHITE), run_time = anispeed)
        self.wait(.6)
        
        drawArr(self, layer2, changeColor, anispeed)
        drawArr(self, layer3, changeColor, anispeed)
        drawArr(self, layer4, changeColor, anispeed)
        drawArr(self, lines23, changeColor, anispeed/2)
        drawArr(self, lines34, changeColor, anispeed/2)
        self.play(Create(output), run_time = anispeed)
        self.wait(1)
        
def drawArr(obj, arr, drawColor, anispeed2):
    for val in arr:
        val.set_color(drawColor)
        obj.play(Create(val), run_time = anispeed2)
        obj.play(val.animate.set_color(WHITE), run_time = anispeed2)

def makeLayer(obj, layer, numHiddens, offset):
    for x in range(0, numHiddens):
        layer.append(Circle())
        layer[x].set_color(WHITE)
        if(x == 0):
            layer[x].shift(DOWN*5+LEFT*offset)
        else:
            layer[x].next_to(prevCirc, UP, buff = 0.5)
        prevCirc = layer[x]

def connectLayers(layer1, layer2):
    lines = []
    for layer in layer1:
        for layere in layer2:
            lines.append(Line(layer, layere))
    return lines

