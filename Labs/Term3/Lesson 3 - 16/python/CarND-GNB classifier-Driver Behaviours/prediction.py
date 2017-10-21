#!/usr/bin/env python

from classifier_solution import GNB
import json


def main():
	gnb = GNB()

	j = json.loads(open('train.json').read())
	print(j.keys())
	X = j['states']
	Y = j['labels']
	gnb.train(X, Y)

	j = json.loads(open('test.json').read())
	X = j['states']
	Y = j['labels']
	score = 0
	for coords, label in zip(X,Y):
		predicted = gnb.predict(coords)
		if predicted == label:
			score += 1
	fraction_correct = float(score) / len(X)
	print("You got {} percent correct".format(100 * fraction_correct))


if __name__ == "__main__":
	main()