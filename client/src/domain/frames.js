import i18n from '../i18n';
import FloralFrame from '../../assets/frames/floral-frame.svg';
import WoodFrame from '../../assets/frames/wood-frame.svg';
import BaroqueFrame from '../../assets/frames/baroque-frame.svg';
import PolaroidFrame from '../../assets/frames/polaroid-frame.svg';
import CalendarFrame from '../../assets/frames/calendar-frame.svg';

// Cosmetic frames earned from milestone trophies (see achievements.js).
// Each SVG (assets/frames/*.svg) is hand-drawn with a transparent "window"
// cut into it for the photo to show through — `viewBox` and `window` here
// mirror that file's own coordinates exactly, so FrameBorder.js can
// position/crop the photo to land precisely inside the cutout rather than
// just stretching a photo behind an unrelated border. If a frame's SVG is
// ever redrawn with a different window, update its entry here to match.
export const FRAMES = [
  {
    id: 'floral-frame',
    label: i18n.t('rewards.frames.floral-frame.label'),
    description: i18n.t('rewards.frames.floral-frame.description'),
    ringColor: '#e8536b',
    icon: 'flower',
    Svg: FloralFrame,
    viewBox: { width: 400, height: 520 },
    window: { x: 95, y: 95, width: 210, height: 330 },
  },
  {
    id: 'wood-frame',
    label: i18n.t('rewards.frames.wood-frame.label'),
    description: i18n.t('rewards.frames.wood-frame.description'),
    ringColor: '#a9713f',
    icon: 'tree',
    Svg: WoodFrame,
    viewBox: { width: 440, height: 540 },
    window: { x: 76, y: 76, width: 288, height: 388 },
  },
  {
    id: 'polaroid-frame',
    label: i18n.t('rewards.frames.polaroid-frame.label'),
    description: i18n.t('rewards.frames.polaroid-frame.description'),
    ringColor: '#e5ddc8',
    icon: 'polaroid',
    Svg: PolaroidFrame,
    viewBox: { width: 440, height: 500 },
    window: { x: 50, y: 50, width: 340, height: 300 },
  },
  {
    id: 'calendar-frame',
    label: i18n.t('rewards.frames.calendar-frame.label'),
    description: i18n.t('rewards.frames.calendar-frame.description'),
    ringColor: '#c9a876',
    icon: 'calendar-month',
    Svg: CalendarFrame,
    viewBox: { width: 440, height: 640 },
    window: { x: 40, y: 50, width: 360, height: 270 },
  },
  {
    id: 'baroque-frame',
    label: i18n.t('rewards.frames.baroque-frame.label'),
    description: i18n.t('rewards.frames.baroque-frame.description'),
    ringColor: '#d4af37',
    icon: 'crown',
    Svg: BaroqueFrame,
    viewBox: { width: 400, height: 500 },
    window: { x: 92, y: 92, width: 216, height: 316 },
  },
];

export function getFrame(frameId) {
  return FRAMES.find((f) => f.id === frameId) || null;
}
