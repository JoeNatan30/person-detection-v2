import math
import numpy as np
import open3d as o3d

"""
Return Acosine of Cos( V1 . V2 / |V1||V2| )
where V1 and V2 are two normalized vectors

Note: These vectors must be normalized
"""


def angleBetweenTwoVectors(v1_normalized, v2_normalized):

    # This dot return cosine of the angle
    cosineOfAngle = np.dot(v1_normalized, v2_normalized)

    if(cosineOfAngle > 1):
        cosineOfAngle = 1
    if(cosineOfAngle < -1):
        cosineOfAngle = -1

    return math.acos(cosineOfAngle)


"""
Return a normalized vector
"""


def vectorNormalization(vector):

    # Norm
    v_norm = np.linalg.norm(vector)

    # Normalization
    return vector/v_norm


"""
Return a normalized vector but norma is of ord = 2
"""


def vectorNormalizationNorm2(vector):

    # Norm
    v_norm2 = np.linalg.norm(vector, ord=2)

    # Normalization
    return vector/v_norm2


"""
Return a value between 0 and 1

mathly the angle of each fpfh component will be less than 2pi
So, it's posible to normilize with this formula:
    (math.cos(  X /2 + math.pi) + 1 ) / 2
"""


def componentsNormalization(fpfh):

    alpha = CosineAngleFormula(fpfh[0])
    phi = CosineAngleFormula(fpfh[1])
    theta = CosineAngleFormula(fpfh[2])

    fpfh[0] = alpha
    fpfh[1] = phi
    fpfh[2] = theta

    return fpfh


"""
formula: (math.cos(  X /2 + math.pi) + 1 ) / 2
"""


def CosineAngleFormula(angle):

    while(angle > math.pi*2):
        angle - (math.pi*2)

    angle = angle/2
    angle += math.pi

    cosine = math.cos(angle)
    cosine += 1
    cosine /= 2

    return cosine


def getKdtree(points):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return o3d.geometry.KDTreeFlann(pcd)

def getEstimatedNormals(pcd, quantity):
    
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=5, max_nn=quantity))
    return pcd

def fpfh(pcd_down, max_nn=100, radius=5):
    
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))

def fixNormalDirectionInPCD(pcd):
    
    normals = np.asarray(pcd.normals)
    points = np.asarray(pcd.points)
    
    long = len(normals)
    
    for pos in range(long):
        suma = 0
        # Punto is used because it's known the viewPoint as (0,0,0)
        vector_view_punto = -1 * points[pos]
    
        for coord in range(3):
            suma = suma + (normals[pos][coord] * vector_view_punto[coord])
    
        if suma >= 0:
            normals[pos] = normals[pos]
        else:
            normals[pos] = -1*normals[pos]
    
    pcd.normals = o3d.utility.Vector3dVector(normals)
    
    return pcd

def showNormals(pcd):

    o3d.visualization.draw_geometries([pcd],
                                      zoom=0.1,
                                      front=[0, 0, 1],
                                      lookat=[0, 0, 0],
                                      up=[0, 1, 0],
                                      point_show_normal=True)
