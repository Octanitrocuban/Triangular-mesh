# Triangular-mesh
A set of function to create regularly spaced triangular mesh

There are two method to crate a mesh, which can be choose in the function **create_mesh**.

All the following actions can be applied on the mesh, regardless of the method used.

There are two method to create maze from the created mesh. The 'fusion' method with **maze_fusion**. The 'exploration' method with **maze_exploration**.
![Exemple picture](circle_methods.png)

![Exemple picture](square_methods.png)

There are two method to solve the created maze. The 'gradient' method with **Dijkstra_triangular_mesh** and **optimal_path** which will compute the shortest path. The 'right' or 'left hand' method with **hand_solving** which will find a path by following a wall.


For the creation of height map, you can applie noise on the node as initial heigth. Then use **lin_smooth** or **kernel_smooth** to smooth this random height.
![Exemple picture](circle_height.png)

![Exemple picture](square_height.png)
