
def mesh_tape_ngsolve(thickness_tape : float = 1e-6,
                     width_tape : float = 4e-3,
                     rbox : float = 20e-3,
                     hmax_tape_corner : float = 1e-6,
                     hmax_tape_inside : float = 10e-6,
                     hmax_out : float = 5e-3,
                     out_label : str = "out",
                     bottom_label : str = "bottom",
                     left_label : str = "left",
                     air_label : str = "air",
                     hts_label : str = "hts"):
    """ Create and returns 2D ngsolve mesh of a tape in air with symmetries. """
    
    from netgen.geom2d import SplineGeometry
    from ngsolve import Mesh

    geo = SplineGeometry()
    
    pnts =[[(0,0), {"maxh": hmax_tape_inside}],
           [(width_tape/2, 0), {"maxh": hmax_tape_corner}],
           [(rbox,0), {"maxh": hmax_out}],
           [(rbox,rbox), {"maxh": hmax_out}],
           [(0,rbox), {"maxh": hmax_out}],
           [(0,thickness_tape/2), {"maxh": hmax_tape_inside}],
           [(width_tape/2,thickness_tape/2), {"maxh": hmax_tape_corner}],
           ]
    
    for pnt, props in pnts:
        geo.AppendPoint(*pnt, **props)
        
    hts = 1
    air = 2
    
    lines = [[["line",0,1], {"leftdomain": hts, "rightdomain": 0, "bc" : bottom_label, "maxh": hmax_tape_inside}],
             [["line",1,2], {"leftdomain": air, "rightdomain": 0, "bc" : bottom_label}],
             [["spline3",2,3,4], {"leftdomain": air, "rightdomain": 0, "maxh": hmax_out, "bc" : out_label}],
             [["line",4,5], {"leftdomain": air, "rightdomain": 0, "bc" : left_label}],
             [["line",5,0], {"leftdomain": hts, "rightdomain": 0, "bc" : left_label}],
             [["line",1,6], {"leftdomain": hts, "rightdomain": air, "maxh": hmax_tape_corner}],
             [["line",5,6], {"leftdomain": air, "rightdomain": hts, "maxh": hmax_tape_inside}],
            ]
    
    for line, props in lines:
        geo.Append(line, **props)
    
    geo.SetMaterial(air, air_label)
    geo.SetMaterial(hts, hts_label)
    
    geo.SetDomainMaxH(1, hmax_tape_inside)
    
    return geo.GenerateMesh(maxh = max([hmax_tape_corner, hmax_tape_inside, hmax_out]))

################################################################################################


import re
import numpy as np

def parse_comsol_mesh_2D(filename):
    """ Read COMSOL mesh file and store data in a dictionary."""
    with open(filename, "r") as f:
        textLines = f.readlines()

    msh = {}
    msh["elts"] = {}
    msh["nodes"] = {}

    flagNode = False
    flagElts = False
    flagInd = False

    for line in textLines:
        line = line.rstrip("\n")

        # -----------------
        # dim
        # -----------------
        if "# sdim" in line:
            m = re.search(r"\d*", line)
            msh["dim"] = int(line[m.start():m.end()])
            dim = int(msh["dim"])

        # -----------------
        # nodes
        # -----------------
        elif "# number of mesh vertices" in line:
            m = re.search(r"\d*", line)
            Nn = int(line[m.start():m.end()])
            msh["nodes"]["N"] = Nn

        elif "# Mesh vertex coordinates" in line:
            flagNode = True
            msh["nodes"]["coordinates"] = np.zeros((Nn, dim))
            conterNode = 0

        elif flagNode:
            m = re.search(r"(-?\d+\.?\d*e?-?\d*\s{1,3}){2}", line)
            if m is None:
                flagNode = False
                x = msh["nodes"]["coordinates"][:, 0]
                y = msh["nodes"]["coordinates"][:, 1]
            else:
                msh["nodes"]["coordinates"][conterNode, :] = \
                    np.fromstring(line[m.start():m.end()], sep=" ")
                conterNode += 1

        # -----------------
        # element type name
        # -----------------
        elif "# type name" in line:
            m = re.search(r"\w+(?= # type name)", line)
            eltsName = line[m.start():m.end()]
            msh["elts"][eltsName] = {}

        # -----------------
        # vertices per element
        # -----------------
        elif "# number of vertices per element" in line:
            m = re.search(r"\d+(?= # number of vertices per element)", line)
            nVertices = int(line[m.start():m.end()])

        # -----------------
        # number of elements
        # -----------------
        elif "# number of elements" in line:
            m = re.search(r"\d+(?= # number of elements)", line)
            nElts = int(line[m.start():m.end()])
            msh["elts"][eltsName]["N"] = nElts
            msh["elts"][eltsName]["connectivity"] = np.zeros(
                (nElts, nVertices), dtype=int
            )

        # -----------------
        # Elements
        # -----------------
        elif "# Elements" in line:
            flagElts = True
            counterElts = 0

        elif flagElts:
            m = re.search(r"(\d+\s)+", line)
            if m is None:
                flagElts = False
            else:
                index = np.fromstring(line[m.start():m.end()], sep=" ", dtype=int)

                xMean = np.mean(x[index ])
                yMean = np.mean(y[index ])

                iSort = np.argsort(
                    np.arctan2(y[index ] - yMean, x[index ] - xMean)
                )

                msh["elts"][eltsName]["connectivity"][counterElts, :] = \
                    index[iSort]
                
                counterElts += 1

        # -----------------
        # geometric entity indices
        # -----------------
        elif "# number of geometric entity indices" in line:
            m = re.search(r"\d+(?= # number of geometric entity indices)", line)
            nInd = int(line[m.start():m.end()])
            msh["elts"][eltsName]["Nind"] = nInd
            msh["elts"][eltsName]["indices"] = np.zeros(nElts, dtype=int)

        elif "# Geometric entity indices" in line:
            flagInd = True
            counterInd = 0

        elif flagInd:
            m = re.search(r"\d+", line)
            if m is None:
                flagInd = False
            else:
                msh["elts"][eltsName]["indices"][counterInd] = \
                    int(line[m.start():m.end()])
                counterInd += 1
    return msh


from netgen.meshing import *
from ngsolve import Mesh as ngsolveMesh

def renameMat(mesh, dicoMat):
    mat = mesh.GetMaterials()
    for i,m in enumerate(mat):
        if m in dicoMat.keys():
            mesh.ngmesh.SetMaterial(i+1,dicoMat[m])
    return mesh

def renameBND(mesh, dicoBND):
    lines = mesh.GetBoundaries()
    for i,line in enumerate(lines):
        if line in dicoBND.keys():
            mesh.ngmesh.SetBCName(i,dicoBND[line])
    return mesh

def import_comsol_mesh_2D(filename, labelBND = {}, labelMat = {}):
    msh = parse_comsol_mesh_2D(filename)
    dim = msh["dim"]
    ngmesh = Mesh(dim=dim)

    # nodes
    Nn = msh["nodes"]["N"]
    nodes = np.zeros([Nn, 3])
    nodes[:,0:dim] = msh["nodes"]["coordinates"]
    pnums = [ngmesh.Add(MeshPoint(Pnt(*nodes[i,:]))) for i in range(Nn)]
    # elements
    for eltsName in msh["elts"].keys():
        Nelt = msh["elts"][eltsName]["N"]
        conn = msh["elts"][eltsName]["connectivity"]
        Nvertices = conn.shape[1]
        allIdx = np.unique(msh["elts"][eltsName]["indices"])
        if Nvertices == 1:
            idx_dom = {str(i) : ngmesh.AddRegion(str(i), dim=0) for i in allIdx}
            for i in range(Nelt):
                index = idx_dom[str(msh["elts"][eltsName]["indices"][i])]
                vertices = pnums[msh["elts"][eltsName]["connectivity"][i,0]]
                ngmesh.Add(Element0D(vertex = vertices,
                                        index = index))
        elif Nvertices == 2:
            idx_dom = {str(i) : ngmesh.AddRegion(str(i), dim=1) for i in allIdx}
            for i in range(Nelt):
                index = idx_dom[str(msh["elts"][eltsName]["indices"][i])]
                vertices = [pnums[v] for v in msh["elts"][eltsName]["connectivity"][i,:]]
                ngmesh.Add(Element1D(vertices = vertices,
                                        index = index))
                
        elif Nvertices >= 3:
            idx_dom = {str(i) : ngmesh.AddRegion(str(i), dim=2) for i in allIdx}
            for i in range(Nelt):
                index = idx_dom[str(msh["elts"][eltsName]["indices"][i])]
                vertices = [pnums[v] for v in msh["elts"][eltsName]["connectivity"][i,:]]
                ngmesh.Add(Element2D(vertices = vertices,
                                        index = index))
    mesh = ngsolveMesh(ngmesh)
    mesh = renameBND(mesh, labelBND)
    mesh = renameMat(mesh, labelMat)     
    return mesh



def parse_comsol_mesh_3d(filename):
    """ Read COMSOL 3D mesh file and store data in a dictionary."""
    with open(filename, "r") as f:
        textLines = f.readlines()

    msh = {}
    msh["elts"] = {}
    msh["nodes"] = {}

    flagNode = False
    flagElts = False
    flagInd = False
    current_elts_name = None

    for line in textLines:
        line = line.rstrip("\n")

        # -----------------
        # dim
        # -----------------
        if "# sdim" in line:
            m = re.search(r"\d+", line)
            msh["dim"] = int(line[m.start():m.end()])
            dim = int(msh["dim"])

        # -----------------
        # nodes
        # -----------------
        elif "# number of mesh vertices" in line:
            m = re.search(r"\d+", line)
            Nn = int(line[m.start():m.end()])
            msh["nodes"]["N"] = Nn

        elif "# Mesh vertex coordinates" in line:
            flagNode = True
            msh["nodes"]["coordinates"] = np.zeros((Nn, 3))  # Always 3D
            counterNode = 0

        elif flagNode:
            # Match 3 coordinates
            m = re.search(r"(-?\d+\.?\d*e?-?\d*\s+){2}-?\d+\.?\d*e?-?\d*", line)
            if m is None:
                flagNode = False
            else:
                coords = np.fromstring(line, sep=" ", count=3)
                # Pad with zeros if 2D data
                msh["nodes"]["coordinates"][counterNode, :len(coords)] = coords
                counterNode += 1

        # -----------------
        # element type name
        # -----------------
        elif "# type name" in line:
            m = re.search(r"\w+(?= # type name)", line)
            if m:
                current_elts_name = line[m.start():m.end()]
                msh["elts"][current_elts_name] = {}

        # -----------------
        # vertices per element
        # -----------------
        elif "# number of vertices per element" in line:
            m = re.search(r"\d+(?= # number of vertices per element)", line)
            nVertices = int(line[m.start():m.end()])

        # -----------------
        # number of elements
        # -----------------
        elif "# number of elements" in line:
            m = re.search(r"\d+(?= # number of elements)", line)
            nElts = int(line[m.start():m.end()])
            msh["elts"][current_elts_name]["N"] = nElts
            msh["elts"][current_elts_name]["connectivity"] = np.zeros(
                (nElts, nVertices), dtype=int
            )

        # -----------------
        # Elements
        # -----------------
        elif "# Elements" in line and current_elts_name:
            flagElts = True
            counterElts = 0

        elif flagElts:
            m = re.search(r"(\d+\s+)+", line)
            if m is None:
                flagElts = False
            else:
                # Parse all integers in the line
                indices = np.fromstring(line, sep=" ", dtype=int)
                if len(indices) >= nVertices:
                    msh["elts"][current_elts_name]["connectivity"][counterElts, :] = indices[:nVertices]
                    counterElts += 1

        # -----------------
        # geometric entity indices
        # -----------------
        elif "# number of geometric entity indices" in line and current_elts_name:
            m = re.search(r"\d+(?= # number of geometric entity indices)", line)
            nInd = int(line[m.start():m.end()])
            msh["elts"][current_elts_name]["Nind"] = nInd
            msh["elts"][current_elts_name]["indices"] = np.zeros(nElts, dtype=int)

        elif "# Geometric entity indices" in line and current_elts_name:
            flagInd = True
            counterInd = 0

        elif flagInd:
            m = re.search(r"\d+", line)
            if m is None:
                flagInd = False
                current_elts_name = None
            else:
                msh["elts"][current_elts_name]["indices"][counterInd] = \
                    int(line[m.start():m.end()])
                counterInd += 1
    return msh


def import_comsol_mesh_3d(filename, labelBND={}, labelMat={}):
    msh = parse_comsol_mesh_3d(filename)
    dim = msh["dim"]
    ngmesh = Mesh(dim=dim)

    # nodes
    Nn = msh["nodes"]["N"]
    nodes = msh["nodes"]["coordinates"]
    pnums = [ngmesh.Add(MeshPoint(Pnt(*nodes[i,:]))) for i in range(Nn)]
    
    # Track which indices correspond to which dimensions
    dim_regions = {0: {}, 1: {}, 2: {}, 3: {}}  # dimension -> {index: region}
    
    # Process elements by type
    for eltsName in msh["elts"].keys():
        Nelt = msh["elts"][eltsName]["N"]
        conn = msh["elts"][eltsName]["connectivity"]
        Nvertices = conn.shape[1]
        indices = msh["elts"][eltsName]["indices"]
        unique_indices = np.unique(indices)
        
        # Determine dimension from element type name
        if "edg" in eltsName.lower() or Nvertices == 2:
            element_dim = 1
        elif "tri" in eltsName.lower() or Nvertices == 3:
            element_dim = 2
        elif "tet" in eltsName.lower() or Nvertices == 4:
            element_dim = 3
        elif Nvertices == 1:
            element_dim = 0
        else:
            print(f"Warning: Unknown element type {eltsName} with {Nvertices} vertices")
            continue
        
        # Create regions for unique indices
        for idx in unique_indices:
            if idx not in dim_regions[element_dim]:
                dim_regions[element_dim][idx] = ngmesh.AddRegion(str(idx), dim=element_dim)
        
        # Add elements
        for i in range(Nelt):
            index = dim_regions[element_dim][indices[i]]
            vertices = [pnums[v] for v in conn[i, :]]
            
            if element_dim == 0:
                ngmesh.Add(Element0D(vertex=vertices[0], index=index))
            elif element_dim == 1:
                ngmesh.Add(Element1D(vertices=vertices, index=index))
            elif element_dim == 2:
                ngmesh.Add(Element2D(vertices=vertices, index=index))
            elif element_dim == 3:
                ngmesh.Add(Element3D(vertices=vertices, index=index))
    
    # Create mesh and rename
    mesh = ngsolveMesh(ngmesh)
    
    # Identify volumes (3D regions)
    #print("Volumes found (3D regions):")
    #for idx, region in dim_regions[3].items():
    #    print(f"  Volume {idx}: region '{ngmesh.GetMaterial(region)}'")
    
    mesh = renameBND(mesh, labelBND)
    mesh = renameMat(mesh, labelMat)     
    return mesh




################################################################################################

def mesh_tape_comsol(quarter = False):
    """ Read and returns 2D comsol mesh """
    
    if quarter:
        labelMat = {
            "1": "hts",
            "2": "air",
        }
        labelBND = {
            "0": "left_hts",
            "1": "bottom_hts",
            "2": "left_air",
            "3": "top_hts",
            "4": "right_hts",
            "5": "bottom_air",
            "6": "out",
        }
        
        try : return import_comsol_mesh_2D( "utils/tape2D_quarter.mphtxt",
                                labelBND=labelBND,
                                labelMat=labelMat)
        
        except: return import_comsol_mesh_2D( "tape2D_quarter.mphtxt",
                                labelBND=labelBND,
                                labelMat=labelMat)
        
    else:
        labelMat = {
            "1": "air",
            "2": "hts",
        }
        labelBND = {
            "0": "left_hts",
            "1": "bottom_hts",
            "2": "top_hts",
            "3": "right_hts",
            "4": "out",
            "5": "out",
            "6": "out",
            "7": "out",
        }

        try : return import_comsol_mesh_2D( "utils/mesh_comsol_2D.mphtxt",
                                labelBND=labelBND,
                                labelMat=labelMat)
        
        except: return import_comsol_mesh_2D( "mesh_comsol_2D.mphtxt",
                                labelBND=labelBND,
                                labelMat=labelMat)


def mesh_bulk_comsol(eighth = False):
    """ Read and returns 3D comsol mesh """
    
    if eighth:
        name = "bulk3D_eighth.mphtxt"
        labelBND = {
            "0": "sym",
            "1": "sym", 
            "2": "bottom_hts", 
            "3": "sym",             
            "4": "sym", 
            "6": "out", 
            "8": "bottom_air", 
            "9": "out", 
            "11": "out",  
        }

        labelMat = {
            "1": "hts",
            "2": "air",
        }
    else:
        name = "mesh_comsol_3D.mphtxt"
        labelBND = {
            "0": "out",  # Dirichlet_X_M
            "1": "out",  # Dirichlet_Y_M
            "2": "out",  # Dirichlet_Z_M
            "11": "out", # Dirichlet_X_P
            "4": "out",  # Dirichlet_Y_P
            "3": "out",  # Dirichlet_Z_P
        }

        labelMat = {
            "1": "air",
            "2": "hts",
        }


    try : return import_comsol_mesh_3d( "utils/" + name,
                             labelBND=labelBND,
                             labelMat=labelMat)
    
    except: return import_comsol_mesh_3d( name,
                             labelBND=labelBND,
                             labelMat=labelMat)

################################################################################################



if __name__ == "__main__":
    mesh2D = mesh_tape_comsol()
    print("import mesh 2D successful")
    mesh3D = mesh_bulk_comsol()
    print("import mesh 3D successful")
    