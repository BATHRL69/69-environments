## Setup

1. Go to Environment testing\Parking\tactics2d\tactics2d\sensor\render_manager.py
2. Update render function to:

```
def render(self):
        """Render the observation of all the sensors."""

        self._clock.tick(self.fps)

        # Handle pygame events to keep the window responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        blit_sequence = []
        for id_, layout_info in self._layouts.items():
            surface = pygame.transform.scale_by(self._sensors[id_].surface, layout_info[0])
            blit_sequence.append((surface, layout_info[1]))

        if self._screen is not None:
            self._screen.blits(blit_sequence)
        pygame.display.flip()
```
