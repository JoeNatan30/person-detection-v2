import math
import numpy as np

"""
Return Acosine of Cos( V1 . V2 / |V1||V2| )
where V1 and V2 are two normalized vectors

Note: These vectors must be normalized
"""
def angleBetweenTwoVectors(v1_normalized,v2_normalized):
    
    #This dot return cosine of the angle
    cosineOfAngle = np.dot(v1_normalized,v2_normalized)
    
    return math.acos(cosineOfAngle)

"""
Return a normalized vector
"""
def vectorNormalization(vector):
    
    #Norm
    v_norm = np.linalg.norm(vector)

    #Normalization
    return vector/v_norm

"""
Return a normalized vector but norma is of ord = 2
"""
def vectorNormalizationNorm2(vector):
    
    #Norm
    v_norm2 = np.linalg.norm(vector*1.0, ord=2)

    #Normalization
    return vector/v_norm2

"""
Return a value between 0 and 1

mathly the angle of each fpfh component will be less than 2pi
So, it's posible to normilize with this formula:
    (math.cos(  X /2 + math.pi) + 1 ) / 2
"""
def componentsNormalization(fpfh):
  
    alpha = CosineAngleFormula(fpfh[0])
    phi   = CosineAngleFormula(fpfh[1])
    theta = CosineAngleFormula(fpfh[2])
    
    fpfh[0] = alpha 
    fpfh[1] = phi
    fpfh[2] = theta
    
    return fpfh

"""
formula: (math.cos(  X /2 + math.pi) + 1 ) / 2
"""
def CosineAngleFormula(angle):
    
    while(angle > math.pi*2): angle - (math.pi*2)
    
    angle = angle/2
    angle += math.pi
    
    cosine = math.cos(angle)
    cosine += 1
    cosine /= 2
        
    return cosine