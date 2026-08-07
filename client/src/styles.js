import { StyleSheet, Dimensions } from 'react-native';

const { width } = Dimensions.get('window');

// Field-guide dark theme
const PALETTE = {
  bg: 'rgb(14, 17, 22)', // very dark slate
  surface: '#0f1720',
  card: '#131722', // slightly lighter card
  cardAlt: '#1b2430',
  cardBorder: '#232c38', // hairline used for elevation instead of drop shadows
  primary: '#1f9d6b', // forest green — the app's one signature color
  accent: '#f59e0b', // amber — reserved for caution/low-confidence states
  danger: '#f44336', // reserved strictly for real errors
  mutedText: '#9aa6b2',
  textOnDark: '#e6eef8',
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: PALETTE.bg,
  },
  mainContent: {
    flex: 1,
  },
  scrollContainer: {
    flexGrow: 1,
    paddingHorizontal: 20,
    paddingTop: 8,
    paddingBottom: 32,
  },
  header: {
    backgroundColor: 'transparent',
    paddingTop: 18,
    paddingBottom: 6,
    paddingHorizontal: 20,
    marginBottom: 8,
  },
  brandRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 4,
  },
  logoMark: {
    width: 40,
    height: 40,
    borderRadius: 12,
    backgroundColor: PALETTE.primary,
    alignItems: 'center',
    justifyContent: 'center',
  },
  brandTextWrap: {
    flex: 1,
    marginLeft: 12,
  },
  brandTitle: {
    fontSize: 20,
    fontWeight: '800',
    color: PALETTE.textOnDark,
    letterSpacing: 0.2,
  },
  brandSubtitle: {
    fontSize: 12,
    color: PALETTE.mutedText,
    marginTop: 2,
  },
  statusIndicator: {
    width: 12,
    height: 12,
    borderRadius: 6,
    marginLeft: 10,
  },
  statusHealthy: {
    backgroundColor: PALETTE.primary,
  },
  statusUnhealthy: {
    backgroundColor: PALETTE.danger,
  },
  healthBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 4,
    marginBottom: 4,
  },
  healthBannerText: {
    color: PALETTE.mutedText,
    fontSize: 12,
    marginLeft: 8,
  },
  healthBannerError: {
    backgroundColor: 'rgba(244, 67, 54, 0.12)',
    borderRadius: 10,
    paddingVertical: 8,
    paddingHorizontal: 12,
    marginTop: 8,
    alignItems: 'center',
  },
  healthBannerErrorText: {
    color: PALETTE.danger,
    fontSize: 13,
    fontWeight: '600',
  },
  imageCard: {
    backgroundColor: PALETTE.cardAlt,
    borderRadius: 12,
    padding: 12,
    marginBottom: 14,
    borderWidth: 1,
    borderColor: PALETTE.cardBorder,
  },
  image: {
    width: width - 96,
    height: width - 240,
    borderRadius: 8,
    resizeMode: 'cover',
  },

  // Home screen hero CTA — the one thing this app does
  heroCard: {
    backgroundColor: PALETTE.cardAlt,
    borderRadius: 24,
    paddingVertical: 32,
    paddingHorizontal: 24,
    alignItems: 'center',
    marginBottom: 24,
    borderWidth: 1,
    borderColor: PALETTE.cardBorder,
  },
  heroIconCircle: {
    width: 72,
    height: 72,
    borderRadius: 36,
    backgroundColor: PALETTE.primary,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 16,
  },
  heroTitle: {
    fontSize: 22,
    fontWeight: '800',
    color: PALETTE.textOnDark,
    marginBottom: 6,
  },
  heroSubtitle: {
    fontSize: 14,
    color: PALETTE.mutedText,
    textAlign: 'center',
    lineHeight: 20,
    marginBottom: 20,
  },
  heroButton: {
    backgroundColor: PALETTE.primary,
    paddingVertical: 14,
    borderRadius: 12,
    width: '100%',
    alignItems: 'center',
  },
  heroButtonText: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: '700',
  },
  sectionLabel: {
    fontSize: 13,
    fontWeight: '700',
    color: PALETTE.mutedText,
    textTransform: 'uppercase',
    letterSpacing: 0.6,
    marginBottom: 10,
  },

  // Not-yet-built menu items — visually demoted, not dead buttons pretending to work
  tileDisabled: {
    backgroundColor: PALETTE.cardAlt,
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    opacity: 0.55,
  },
  tileLeft: { flexDirection: 'row', alignItems: 'center' },
  tileIconDisabled: {
    width: 54,
    height: 54,
    borderRadius: 14,
    backgroundColor: 'rgba(154, 166, 178, 0.12)',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 12,
  },
  tileTextDisabled: { color: PALETTE.mutedText, fontSize: 16, fontWeight: '700' },
  tileSub: { color: PALETTE.mutedText, fontSize: 13 },
  soonBadge: {
    backgroundColor: PALETTE.surface,
    borderRadius: 8,
    paddingHorizontal: 8,
    paddingVertical: 4,
  },
  soonBadgeText: {
    color: PALETTE.mutedText,
    fontSize: 11,
    fontWeight: '700',
    textTransform: 'uppercase',
    letterSpacing: 0.4,
  },

  // Nested inside resultCard — no border of its own, grouped by fill only
  predictionsCard: {
    backgroundColor: PALETTE.cardAlt,
    borderRadius: 16,
    padding: 12,
    marginBottom: 16,
  },
  predictionsTitle: {
    color: PALETTE.textOnDark,
    fontSize: 15,
    fontWeight: '700',
    marginBottom: 12,
  },
  predictionRow: {
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#111a24',
  },
  predictionRowLast: {
    borderBottomWidth: 0,
  },
  predictionRowTop: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 6,
  },
  predictionRank: {
    width: 30,
    height: 30,
    borderRadius: 10,
    backgroundColor: '#0b1620',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 12,
  },
  predictionRankText: {
    color: PALETTE.textOnDark,
    fontWeight: '700',
  },
  predictionLabel: {
    flex: 1,
    color: PALETTE.textOnDark,
    fontSize: 15,
  },
  predictionBarTrack: {
    height: 4,
    borderRadius: 2,
    backgroundColor: PALETTE.cardBorder,
    overflow: 'hidden',
  },
  predictionBarFill: {
    height: '100%',
    borderRadius: 2,
    backgroundColor: PALETTE.primary,
  },

  bottomBrand: {
    alignItems: 'center',
    marginTop: 28,
    paddingVertical: 18,
  },

  placeholderContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingVertical: 40,
  },
  placeholderCard: {
    backgroundColor: PALETTE.card,
    borderRadius: 12,
    padding: 18,
    alignItems: 'flex-start',
    marginBottom: 14,
  },
  placeholderIcon: {
    fontSize: 72,
    marginBottom: 18,
    opacity: 0.85,
  },
  placeholderTitle: {
    fontSize: 26,
    fontWeight: '700',
    color: '#ffffff',
    textAlign: 'center',
    marginBottom: 12,
  },
  placeholderText: {
    fontSize: 15,
    color: '#b9c6dc',
    textAlign: 'center',
    lineHeight: 24,
    marginBottom: 22,
  },
  placeholderHint: {
    backgroundColor: '#2a2a2a',
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderRadius: 25,
    borderWidth: 1,
    borderColor: '#444',
  },
  placeholderHintText: {
    color: '#ffffff',
    fontSize: 14,
    fontWeight: '500',
    textAlign: 'center',
  },
  errorCard: {
    backgroundColor: PALETTE.card,
    borderRadius: 12,
    padding: 20,
    marginBottom: 14,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: PALETTE.cardBorder,
  },
  errorTitle: {
    color: PALETTE.danger,
    fontSize: 16,
    fontWeight: '700',
    marginBottom: 6,
  },
  errorText: {
    color: PALETTE.mutedText,
    fontSize: 14,
    textAlign: 'center',
    marginBottom: 16,
  },
  errorButtonRow: {
    flexDirection: 'row',
    width: '100%',
  },
  resultFooter: {
    flexDirection: 'row',
    paddingHorizontal: 20,
    paddingTop: 12,
    paddingBottom: 12,
    backgroundColor: 'transparent',
    borderTopWidth: 1,
    borderTopColor: PALETTE.cardBorder,
  },
  loadingCard: {
    backgroundColor: PALETTE.card,
    borderRadius: 12,
    padding: 20,
    marginBottom: 14,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: PALETTE.cardBorder,
  },
  loadingText: {
    color: '#ffffff',
    fontSize: 18,
    fontWeight: '600',
    marginTop: 15,
  },
  loadingSubtext: {
    color: '#b0b0b0',
    fontSize: 14,
    marginTop: 5,
  },
  resultCard: {
    backgroundColor: 'transparent',
    padding: 14,
    marginHorizontal: -20,
    marginBottom: 14,
  },
  resultCardFill: {
    flex: 1,
  },
  resultImageContainer: {
    marginBottom: 22,
    alignItems: 'center',
  },
  resultHeaderRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  confidenceBadge: {
    backgroundColor: PALETTE.primary,
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 20,
  },
  confidenceText: {
    color: '#ffffff',
    fontSize: 14,
    fontWeight: '700',
  },
  confidenceTextLow: {
    color: PALETTE.bg,
  },
  confidenceBadgeLow: {
    backgroundColor: PALETTE.accent,
  },
  lowConfidenceBanner: {
    backgroundColor: 'rgba(245, 158, 11, 0.14)',
    borderRadius: 10,
    padding: 12,
    marginBottom: 16,
  },
  lowConfidenceText: {
    color: PALETTE.accent,
    fontSize: 13,
    lineHeight: 18,
  },
  uncertainHeaderRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    marginBottom: 6,
  },
  uncertainHeaderTitle: {
    fontSize: 22,
    fontWeight: '800',
    color: PALETTE.textOnDark,
  },
  uncertainHeaderSubtitle: {
    color: PALETTE.mutedText,
    fontSize: 13,
    lineHeight: 18,
    marginBottom: 12,
  },
  uncertainImageWrapper: {
    flex: 1,
    width: '100%',
    borderRadius: 14,
    overflow: 'hidden',
    backgroundColor: PALETTE.surface,
  },
  uncertainImage: {
    width: '100%',
    height: '100%',
    resizeMode: 'cover',
  },
  birdName: {
    flexShrink: 1,
    fontSize: 24,
    fontWeight: '800',
    color: PALETTE.textOnDark,
    marginRight: 12,
  },

  // Species reference lookup (real photo + stats from iNaturalist)
  referenceLoading: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 20,
  },
  referenceLoadingText: {
    color: PALETTE.mutedText,
    fontSize: 13,
    marginLeft: 8,
  },
  referenceErrorCard: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: PALETTE.cardAlt,
    borderRadius: 12,
    padding: 12,
    marginBottom: 20,
  },
  referenceErrorText: {
    flex: 1,
    color: PALETTE.mutedText,
    fontSize: 12,
    marginRight: 12,
  },
  referenceErrorRetry: {
    color: PALETTE.primary,
    fontSize: 12,
    fontWeight: '700',
  },
  referenceCard: {
    backgroundColor: PALETTE.cardAlt,
    borderRadius: 16,
    padding: 12,
    marginBottom: 20,
  },
  subImageRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 10,
    gap: 12,
  },
  subImageColumn: {
    flex: 1,
    alignItems: 'center',
  },
  subImageWrapper: {
    width: '100%',
    aspectRatio: 1.12,
    borderRadius: 14,
    overflow: 'hidden',
    backgroundColor: PALETTE.surface,
    marginBottom: 8,
  },
  subImage: {
    width: '100%',
    height: '100%',
    resizeMode: 'cover',
  },
  subImageLabel: {
    color: PALETTE.mutedText,
    fontSize: 12,
    textTransform: 'uppercase',
  },
  referenceAttribution: {
    color: PALETTE.mutedText,
    fontSize: 11,
    marginBottom: 16,
  },
  galleryRow: {
    flexDirection: 'row',
    gap: 8,
    marginBottom: 16,
  },
  galleryThumb: {
    flex: 1,
    aspectRatio: 1,
    borderRadius: 10,
    backgroundColor: PALETTE.surface,
  },
  galleryThumbImage: {
    width: '100%',
    height: '100%',
    borderRadius: 10,
  },
  referenceStatsGrid: {
    marginBottom: 12,
  },
  referenceStatItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: PALETTE.cardBorder,
  },
  referenceStatItemLast: {
    borderBottomWidth: 0,
  },
  referenceStatLabel: {
    color: PALETTE.mutedText,
    fontSize: 13,
  },
  referenceStatValue: {
    color: PALETTE.textOnDark,
    fontSize: 13,
    fontWeight: '700',
  },
  referenceStatStatus: {
    color: PALETTE.accent,
  },
  rangeSection: {
    marginBottom: 12,
  },
  rangeRow: {
    marginBottom: 8,
  },
  rangeLabel: {
    color: PALETTE.mutedText,
    fontSize: 11,
    textTransform: 'uppercase',
    letterSpacing: 0.4,
    marginBottom: 2,
  },
  rangeValue: {
    color: PALETTE.textOnDark,
    fontSize: 13,
    lineHeight: 18,
  },
  referenceSummary: {
    color: PALETTE.mutedText,
    fontSize: 13,
    lineHeight: 19,
    marginBottom: 8,
  },
  wikiLink: {
    color: PALETTE.primary,
    fontSize: 13,
    fontWeight: '700',
  },

  birdClass: {
    fontSize: 14,
    color: PALETTE.mutedText,
    fontStyle: 'normal',
  },
  modelInfo: {
    backgroundColor: 'transparent',
    padding: 8,
    borderRadius: 8,
    marginBottom: 6,
  },
  modelLabel: {
    fontSize: 12,
    color: PALETTE.mutedText,
    marginBottom: 4,
    letterSpacing: 0.2,
  },
  modelName: {
    fontSize: 14,
    color: PALETTE.textOnDark,
    fontWeight: '600',
  },
  resultMetaGrid: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: 12,
  },
  resultMetaItem: {
    flex: 1,
    backgroundColor: PALETTE.cardAlt,
    borderRadius: 14,
    paddingVertical: 12,
    paddingHorizontal: 14,
  },
  resultMetaLabel: {
    color: PALETTE.mutedText,
    fontSize: 12,
    marginBottom: 4,
  },
  resultMetaValue: {
    color: PALETTE.textOnDark,
    fontSize: 15,
    fontWeight: '700',
  },
  resultActionButton: {
    flex: 1,
    backgroundColor: PALETTE.primary,
    borderRadius: 14,
    paddingVertical: 14,
    alignItems: 'center',
    marginRight: 8,
  },
  resultActionText: {
    color: PALETTE.card,
    fontWeight: '700',
  },
  resultActionButtonSecondary: {
    flex: 1,
    backgroundColor: PALETTE.cardAlt,
    borderRadius: 14,
    paddingVertical: 14,
    alignItems: 'center',
  },
  resultActionTextSecondary: {
    color: PALETTE.textOnDark,
    fontWeight: '700',
  },
  actionPanel: {
    position: 'absolute',
    bottom: 18,
    left: 20,
    right: 20,
    backgroundColor: 'transparent',
    paddingHorizontal: 0,
    paddingVertical: 0,
    alignItems: 'center',
  },
  mainActionButton: {
    backgroundColor: PALETTE.primary,
    paddingVertical: 16,
    paddingHorizontal: 20,
    borderRadius: 12,
    alignItems: 'center',
    width: '100%'
  },
  mainActionText: {
    color: PALETTE.card,
    fontSize: 18,
    fontWeight: '700',
  },
  mainActionSubtext: {
    color: PALETTE.card,
    fontSize: 13,
    marginTop: 4,
    opacity: 0.95,
  },

  // Full-screen image viewer
  lightboxBackdrop: {
    flex: 1,
    backgroundColor: 'rgba(5, 7, 10, 0.94)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  lightboxImage: {
    width: '92%',
    height: '75%',
  },
  lightboxCloseButton: {
    position: 'absolute',
    top: 56,
    right: 20,
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: 'rgba(255, 255, 255, 0.12)',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 1,
  },

  PALETTE,
});

export default styles;
