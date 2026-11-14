# UFC 6 launcher and FG verification, 2026-11-14

The launcher compatibility preset for UFC 6 now emits a 60 Hz virtual vblank and
synchronous shader/pipeline compilation. It retains the existing 1080p, Immediate
presentation and shader-cache recommendations. Backend, factor, FG enable state and
notice-screen choice remain as saved; disabling recommended settings restores the
requested timing/compilation options. The preset changes no saved values or other games.

The duplicate right-pane category heading has also been removed. Windows now reports
immediate process-start failures using the same check already used on other platforms.

## Runtime evidence and limits

Host: RTX 3080, i7-12700K. Game: PPSA23566 v01.004.000. Runtime tested:
45e36998, executable SHA256
`909A77E2A11EAB2F5E14BD736BE4FCA229E9C10C090CEE6A732969DA1671F216`.
These launcher changes do not modify emulator rendering code.

The visually verified main menu was tested with Multi 6x, shader cache, redzone
protection, 60 Hz and synchronous compilation. The existing opt-in test harness sent
one 300 ms Cross pulse at the title and each offline notice; no gameplay inputs were sent.
FG was then toggled off using F3 in the same menu.

- FG 6x: 12 complete diagnostic windows, 24.048 seconds; 20.792 real presented
  frames/s, 91.400 generated frames/s, 112.192 successful real/generated present
  calls/s. No generation failures, failed presents or blank presents were counted.
- FG off: 17 complete windows, 34.080 seconds; 25.851 real presented frames/s,
  no generated frames. No failed or blank presents were counted.

These are application presentation counters, not measured display scanout. FG
consumes GPU time and did not increase real game FPS. This is a menu test, not
proof of match/gameplay compatibility or 60 FPS emulation.

## Earlier crash

Windows recorded integer divide-by-zero (0xc0000094) at guest address
0x906b66c59. The instruction divides by a value derived from a guest object's
size/count; that value was zero. The saved dump omitted the relevant guest stack
and object memory, so the producing call/input could not be established.

A run using 360 Hz and asynchronous settings remained on the initial splash.
The synchronous 60 Hz baseline reached the menu and survived the FG on/off test.
Further 60 Hz asynchronous startup tests, including a separate empty-cache working
directory with copied saves, did not reproduce the divide-by-zero. This rules out
claiming a confirmed FG exception or a proven root-cause crash fix. The preset is a
conservative launch workaround; the cause of the earlier exception remains unconfirmed.

No divide exception was suppressed and no guest binary was patched.
