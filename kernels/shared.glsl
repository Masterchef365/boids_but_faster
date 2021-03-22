struct Boid {
    vec3 pos;
    uint level;
    vec3 heading;
    uint mask;
};

struct AccumulatorHalf {
    vec3 pos;
    uint count;
    vec3 heading;
    uint _filler;
};

struct Accumulator {
    AccumulatorHalf left;
    AccumulatorHalf right;
};
