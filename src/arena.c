#include "../headers/arena.h"

Arena *arenaAlloc(u64 capacity) {
    Arena *arena = malloc(capacity);

    arena->capacity = capacity;
    arena->pos = sizeof(Arena);

    return arena;
}

void arenaFree(Arena *a) {
    free(a);
}

void *pushArena(Arena *a, u64 size) {
    u64 posAligned = ALIGN_UP_POW2(a->pos, ARENA_ALIGN);
    u64 newPos = posAligned + size;

    if (newPos > a->capacity) {
        fprintf(stderr, "Need more space in arenas!!!\n");
        exit(1);
    }

    a->pos = newPos;

    u8 *out = (u8 *)a + posAligned;

    return out;
}

void *pushArenaZero(Arena *a, u64 size) {
    u8 *out = pushArena(a, size);
    memset(out, 0, size);

    return out;
}

void arenaPop(Arena *a, u64 size) {
    size = MIN(size, a->pos - sizeof(Arena));
    a->pos -= size;
}

void arenaPopTo(Arena *a, u64 pos) {
    u64 size = pos < a->pos ? a->pos - pos : 0;
    arenaPop(a, size);
}

void clearArena(Arena *a) {
    arenaPopTo(a, sizeof(Arena));
}
