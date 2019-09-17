#pragma once

/*
Functions for sampling
*/

__host__ __device__
glm::vec3 squareToDiskConcentric(const glm::vec2 &sample) {
    // Used Peter Shirley's concentric disk warp
    float radius;
    float angle;
    float a = (2 * sample[0]) - 1;
    float b = (2 * sample[1]) - 1;

    if (a > -b) {
        if (a > b) {
            radius = a;
            angle = (PI / 4.f) * (b / a);
            }
        else {
            radius = b;
            angle = (PI / 4.f) * (2 - (a / b));
            }
        }
    else {
        if (a < b) {
            radius = -a;
            angle = (PI / 4.f) * (4 + (b / a));
            }
        else {
            radius = -b;
            if (b != 0) {
                angle = (PI / 4.f) * (6 - (a / b));
                }
            else {
                angle = 0;
                }
            }
        }
    return glm::vec3(radius * glm::cos(angle), radius * glm::sin(angle), 0);
}

__host__ __device__
glm::vec3 squareToHemisphereCosine(const glm::vec2 &sample) {
    // Used Peter Shirley's cosine hemisphere
    glm::vec3 disk = squareToDiskConcentric(sample);
    return glm::vec3(disk[0], disk[1], glm::sqrt(1.f - glm::pow(glm::length(disk), 2.f)));
}