// SDL2 Hello, World!
// This should display a white screen for 2 seconds
// compile with: clang++ main.cpp -o hello_sdl2 -lSDL2
// run with: ./hello_sdl2
#include <SDL2/SDL.h>
#include <stdio.h>
#include <iostream>

#include <string.h>

#include "const.h"
#include "gpu.h"

void render(SDL_Surface* screen, SDL_Surface* cuda) {

	const unsigned int color_red   = 0xffff0000;
	const unsigned int color_green = 0xff00ff00;
	const unsigned int color_blue  = 0xff0000ff;
	const unsigned int color_white  = 0xffffffff;

	unsigned int* p = (unsigned int*) screen->pixels;
	int i = 0;

	for(; i < SCREEN_WIDTH * SCREEN_HEIGHT; i++){
		if (i / SCREEN_WIDTH < 120) {
			p[i] = color_red;
		}
		else if( i / SCREEN_WIDTH < 240 ) {
			p[i] = color_green;
		}
		else if( i / SCREEN_WIDTH < 360 ) {
			p[i] = color_blue;
		}
		else {
			p[i] = color_white;
		}
	}
}

int main(int argc, char* args[]) {

	uint32_t time_step = 1000. / 60. ;
	uint32_t next_time_step = SDL_GetTicks();


	if (SDL_Init(SDL_INIT_EVERYTHING) < 0) {
		std::cerr << "could not initialize sdl2: " << SDL_GetError() << std::endl;
		return 1;
	}

	SDL_Window* window = SDL_CreateWindow(
				"main",
				SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
				SCREEN_WIDTH, SCREEN_HEIGHT,
				SDL_WINDOW_SHOWN
				);

	SDL_Surface* default_screen = SDL_CreateRGBSurface( 0, SCREEN_WIDTH, SCREEN_HEIGHT, 32,
												0x00FF0000,
												0x0000FF00,
												0x000000FF,
												0xFF000000);

	SDL_Surface* cuda_screen = SDL_CreateRGBSurface( 0, SCREEN_WIDTH, SCREEN_HEIGHT, 32,
												0x00FF0000,
												0x0000FF00,
												0x000000FF,
												0xFF000000);

	if (default_screen == NULL || cuda_screen == NULL) {
        SDL_Log("SDL_CreateRGBSurface() failed: %s", SDL_GetError());
        exit(1);
	}

    SDL_Renderer* sdlRenderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);

	SDL_Texture *sdlTexture = SDL_CreateTexture(sdlRenderer,
												SDL_PIXELFORMAT_ARGB8888,
												//SDL_TEXTUREACCESS_STREAMING | SDL_TEXTUREACCESS_TARGET,
												SDL_TEXTUREACCESS_TARGET,
												640, 480);

	if (sdlTexture== NULL) {
        SDL_Log("SDL_Error failed: %s", SDL_GetError());
        exit(1);
	}
	

    while (1) {

        SDL_Event e;
        if (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) {
                break;
            }
        }

		uint32_t now = SDL_GetTicks();
		if (next_time_step <= now) {

			SDL_LockSurface(default_screen);
			render(default_screen, cuda_screen);
			SDL_UnlockSurface(default_screen);

			SDL_UpdateTexture(sdlTexture, NULL, default_screen->pixels, default_screen->pitch);
			SDL_RenderClear(sdlRenderer);
			SDL_RenderCopy(sdlRenderer, sdlTexture, NULL, NULL);
			SDL_RenderPresent(sdlRenderer);

			next_time_step += time_step;
		} else {
			SDL_Delay(next_time_step - now);
		}

    }

  if (window == NULL) {
	  std::cerr << "could not create window: " << SDL_GetError() << std::endl;
    return 1;
  }

    SDL_DestroyTexture(sdlTexture);
    SDL_DestroyRenderer(sdlRenderer);
    SDL_DestroyWindow(window);

    SDL_Quit();

  return 0;
}
