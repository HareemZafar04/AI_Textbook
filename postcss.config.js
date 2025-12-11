module.exports = {
  plugins: [
    require('autoprefixer'),
    // Use cssnano but without the problematic SVGO plugin
    [
      require('cssnano'),
      {
        preset: [
          'default',
          {
            svgo: false, // Disable SVGO to avoid the cleanupIds.js error
          },
        ],
      },
    ],
  ],
};