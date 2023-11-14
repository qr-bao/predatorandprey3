import numpy as np
import pygame
import math
import predator
import creature
import constants
# Position               => a tuple of 2D coordinates
# ListOfCounterCreatures => The list which is to be used to get creatures in the field of view
# Upperbound             => the field of view

def PreyFilterUsingEuclideanDistances(Position, ListOfCounterCreatures, Upperbound):
    response = []
    if isinstance(ListOfCounterCreatures, list):
        for animal in ListOfCounterCreatures:
            x,y = animal.rect.centerx, animal.rect.centery
            verctor1 = (x-Position[0],y-Position[1])
            verctor2 = Position
            direction = creature.velocity.normalize()
            distance = ((x-Position[0])**2 + (y-Position[1])**2)**(0.5)
            theta_rad, theta_deg,position = calculate_angle_between_vectors(verctor1, direction)
            if distance < Upperbound and theta_deg <= constants.predatorsViewingAngle:
                if distance != 0:
                    response.append((animal,(1/distance)))
                else:
                    response.append((animal, float('inf')))
    else:
        print("Error: ListOfCounterCreatures argument is not iterable")
    return response
def PreyFilterUsingEuclideanDistances1(Vector,Position, ListOfCounterCreatures, Upperbound):
    response = []
    if isinstance(ListOfCounterCreatures, list):
        for animal in ListOfCounterCreatures:
            x,y = animal.rect.centerx, animal.rect.centery
            verctor1 = (x-Position[0],y-Position[1])
            verctor2 = Vector
            distance = ((x-Position[0])**2 + (y-Position[1])**2)**(0.5)
            theta_rad, theta_deg,position = calculate_angle_between_vectors(verctor1, verctor2)
            if distance < Upperbound and theta_deg <= constants.PredatorsVIEW_RANGE:
                if distance != 0:
                    response.append((animal,(1/distance)))
                else:
                    response.append((animal, float('inf')))
    else:
        print("Error: ListOfCounterCreatures argument is not iterable")
    return response

def PredictPredatorDirection(Position, CreaturesAround, behaviourRate):
    if len(CreaturesAround)==0:
        return pygame.math.Vector2(0, 0)

    desiredVelocitylist = []

    for details in CreaturesAround:
        animal, factor = details
        desiredVelocitylist.append([behaviourRate*factor*(animal.rect.x - Position[0]) , behaviourRate*factor*(animal.rect.y - Position[1])])

    desiredVelocities = np.array(desiredVelocitylist)
    resultant = np.sum(desiredVelocities, axis = 0)
    resultantVector2 = pygame.math.Vector2(resultant[0], resultant[1])
    
    target = CreaturesAround[0][0]
    maxFactor = CreaturesAround[0][1]
    for vel, details in zip(desiredVelocities, CreaturesAround):
        animal, factor = details
        currVector = pygame.math.Vector2(vel[0], vel[1])
        if abs(resultantVector2.angle_to(currVector)) <= 45 and maxFactor < factor:
            maxFactor = factor
            target = animal   

    desired = (pygame.math.Vector2(target.rect.x, target.rect.y) - pygame.math.Vector2(Position))*behaviourRate
    return desired

def FoodAndPredatorFilterUsingEuclideanDistances(vector,Position, ListOfCounterCreatures, ListOfFood, Upperbound):
    preds = []
    foods = []
    creature_positions = []
    for creature in ListOfCounterCreatures:
        x,y = creature.rect.centerx, creature.rect.centery
        distance = ((x-Position[0])**2 + (y-Position[1])**2)**(0.5)
        verctorOfPreyAndPredator = (x-Position[0],y-Position[1])
        velocityDerection = vector
        #position = 0 means the prey is on the left back of the predator
        #position = 1 means the prey is on the right back of the predator
        theta_rad, theta_deg,position = calculate_angle_between_vectors(verctorOfPreyAndPredator, velocityDerection)
        if distance < Upperbound and theta_deg <= constants.PreysVIEW_RANGE:
            if distance != 0 :
                preds.append((creature, (1/distance)))
            else:
                preds.append((creature, float('inf')))
        if distance < Upperbound and theta_deg >= constants.PreysVIEW_RANGE:
            if position ==0:
                creature_positions.append((creature, float(1000)))
            else:
                creature_positions.append((creature, float(2000)))

    
    for food in ListOfFood:
        x, y = food.x, food.y
        distance = ((x-Position[0])**2 + (y-Position[1])**2)**(0.5)
        verctorOfPreyAndFood = (x-Position[0],y-Position[1])
        velocityDerection = vector
        theta_rad, theta_deg,position = calculate_angle_between_vectors(verctorOfPreyAndPredator, velocityDerection)
        if distance < Upperbound and theta_deg <= constants.PreysVIEW_RANGE:
            if distance != 0:
                foods.append((food,(1/distance)))
            else:
                foods.append((food, float('inf')))
    
    return foods, preds

def PredictPreyDirection(Position, CreaturesAround, FoodAround, foodBehaviour, creatureBehaviour):
    if len(CreaturesAround) == 0 and len(FoodAround) == 0:
        return pygame.math.Vector2(0, 0)

    desiredVelocityCreaturelist = [[0,0]]
    for details in CreaturesAround:
        animal, factor = details
        if factor ==1000:
            desiredVelocityCreaturelist.append([creatureBehaviour*factor*(-1 - Position[0]) , creatureBehaviour*factor*(1 - Position[1])])
        if factor ==2000:
            desiredVelocityCreaturelist.append([creatureBehaviour*factor*(1 - Position[0]) , creatureBehaviour*factor*(-1 - Position[1])])
        else:
            desiredVelocityCreaturelist.append([creatureBehaviour*factor*(animal.rect.x - Position[0]) , creatureBehaviour*factor*(animal.rect.y - Position[1])])
    
    desiredVelocityFoodlist = [[0, 0]]
    for details in FoodAround:
        food, factor = details
        desiredVelocityFoodlist.append([foodBehaviour*factor*(food.x - Position[0]), foodBehaviour*factor*(food.y - Position[1])])

    desiredVelocitiesCreature = np.array(desiredVelocityCreaturelist)
    desiredVelocitiesFood = np.array(desiredVelocityFoodlist)
    resultant1 = np.sum(desiredVelocitiesCreature, axis = 0)
    resultant2 = np.sum(desiredVelocitiesFood, axis = 0)

    resultantVector2 = pygame.math.Vector2(resultant1[0] + resultant2[0], resultant1[1] + resultant2[1])

    if(len(CreaturesAround) != 0):
        target = CreaturesAround[0][0]
        maxFactor = CreaturesAround[0][1]
    else:
        target = FoodAround[0][0]
        maxFactor = FoodAround[0][1]

    for vel, details in zip(desiredVelocitiesCreature, CreaturesAround):
        animal, factor = details
        currVector = pygame.math.Vector2(vel[0], vel[1])
        if abs(resultantVector2.angle_to(currVector)) <= 45 and maxFactor < factor:
            maxFactor = factor
            target = animal
    
    for vel, details in zip(desiredVelocitiesFood, FoodAround):
        food, factor = details
        currVector = pygame.math.Vector2(vel[0], vel[1])
        if abs(resultantVector2.angle_to(currVector)) <= 45 and maxFactor < factor:
            maxFactor = factor
            target = food

    if type(target) == predator.Predator:
        return (pygame.math.Vector2(target.rect.x, target.rect.y) - pygame.math.Vector2(Position))*creatureBehaviour
    
    else:
        return (pygame.math.Vector2(target.x, target.y) - pygame.math.Vector2(Position))*foodBehaviour
"""def calculate_angle_between_vectors(vector1, vector2):

    magnitude1 = math.sqrt(vector1[0]**2 + vector1[1]**2)
    magnitude2 = math.sqrt(vector2[0]**2 + vector2[1]**2)
    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
    cosine_theta = dot_product / (magnitude1 * magnitude2)
    cosine_theta = max(-1, min(1, cosine_theta))
    theta_rad = math.acos(cosine_theta)
    theta_deg = math.degrees(theta_rad)
    

    return theta_rad, theta_deg"""
import pygame
import math

def calculate_angle_between_vectors(vector1, vector2):
    # Convert lists to Vector2 objects
    v1 = pygame.math.Vector2(vector1)
    v2 = pygame.math.Vector2(vector2)

    # Calculate the dot product
    dot_product = v1.dot(v2)

    # Calculate the magnitudes of the vectors
    magnitude_v1 = v1.length()
    magnitude_v2 = v2.length()
    if magnitude_v1 * magnitude_v2 == 0:
        cos_theta = 1
    else:
        cos_theta = dot_product / (magnitude_v1 * magnitude_v2)
    # Calculate the cosine of the angle

    # Clamp the value to the range [-1, 1] to avoid any floating point errors
    cos_theta = max(-1, min(1, cos_theta))

    # Use the arccos function to find the angle in radians
    theta_rad = math.acos(cos_theta)

    # Convert the angle to degrees
    theta_deg = math.degrees(theta_rad)

    # Calculate the cross product
    cross_product = v1.x * v2.y - v1.y * v2.x

    # Determine which vector is on the left
    if cross_product > 0:
        position = 0
    elif cross_product < 0:
        position = 1
    else:
        position = 0

    return theta_rad, theta_deg,position