#ifndef PHYSICS_H
#define PHYSICS_H

struct particle {

    // Position
    float x;
    float y;

    // Velocity
    float v_x;
    float v_y;

};

/*
 * Calculates the center of mass of all of the given particles, and sets the given particle
 * "center_of_mass" as the answer
 *
 * Note: mass is currently not a thing
*/
void calculate_center_of_mass(struct particle* center_of_mass, struct particle* particles, int count) {
    float x = 0;
    float y = 0;

    for(int i = 0; i < count; i++) {
        x += particles[i].x;
        y += particles[i].y;
    }

    x /= count;
    y /= count;

    center_of_mass->x = x;
    center_of_mass->y = y;


}

#endif