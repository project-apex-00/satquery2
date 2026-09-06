// interactions.js
// Pure front-end presentation layer: scroll progress, parallax, section
// wayfinding, card tilt, magnetic buttons, cursor spotlight, and
// scroll-triggered reveals. Reads the DOM but never touches apiBaseUrl,
// fetch calls, or any state owned by app.js — safe to load alongside it.

(function () {
  const reduceMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const fine = window.matchMedia && window.matchMedia("(pointer: fine)").matches;

  document.addEventListener("DOMContentLoaded", () => {
    setupScrollProgress();
    setupSectionDots();
    setupHeroParallax();
    setupRevealOnScroll();
    if (!reduceMotion && fine) {
      setupCursorGlow();
      setupCardTilt();
      setupMagnetic();
    }
  });

  // ---------- scroll progress bar ----------
  function setupScrollProgress() {
    const bar = document.getElementById("scroll-progress-fill");
    if (!bar) return;
    let ticking = false;
    const update = () => {
      const doc = document.documentElement;
      const scrollTop = window.scrollY || doc.scrollTop;
      const height = doc.scrollHeight - doc.clientHeight;
      const pct = height > 0 ? (scrollTop / height) * 100 : 0;
      bar.style.width = `${pct}%`;
      ticking = false;
    };
    window.addEventListener(
      "scroll",
      () => {
        if (!ticking) {
          requestAnimationFrame(update);
          ticking = true;
        }
      },
      { passive: true }
    );
    update();
  }

  // ---------- floating section wayfinder ----------
  function setupSectionDots() {
    const nav = document.getElementById("section-dots");
    if (!nav) return;
    const dots = Array.from(nav.querySelectorAll(".dot"));
    const targets = dots
      .map((d) => document.getElementById(d.getAttribute("data-target")))
      .filter(Boolean);

    if (targets.length === 0) return;

    requestAnimationFrame(() => nav.classList.add("is-visible"));

    dots.forEach((dot) => {
      dot.addEventListener("click", () => {
        const target = document.getElementById(dot.getAttribute("data-target"));
        if (target) target.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });
      });
    });

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            dots.forEach((d) => d.classList.remove("active"));
            const match = dots.find((d) => d.getAttribute("data-target") === entry.target.id);
            if (match) match.classList.add("active");
          }
        });
      },
      { rootMargin: "-40% 0px -55% 0px", threshold: 0 }
    );

    targets.forEach((t) => observer.observe(t));
  }

  // ---------- hero cinematic parallax + scroll-cue ring ----------
  function setupHeroParallax() {
    const hero = document.getElementById("hero-section");
    const globeWrap = document.querySelector(".space-globe-wrap");
    const stars = document.querySelector(".space-stars");
    const heroCopy = document.querySelector(".hero-copy");
    const scrollCue = document.getElementById("scroll-cue");
    if (!hero) return;

    let ticking = false;
    const update = () => {
      const heroHeight = hero.offsetHeight || window.innerHeight;
      const progress = Math.min(Math.max(window.scrollY / heroHeight, 0), 1);

      if (!reduceMotion) {
        if (globeWrap) {
          globeWrap.style.transform = `translateY(${progress * 40}px) scale(${1 + progress * 0.06})`;
        }
        if (stars) {
          stars.style.transform = `translateY(${progress * -20}px)`;
        }
        if (heroCopy) {
          heroCopy.style.opacity = String(1 - progress * 1.15);
          heroCopy.style.transform = `translateY(${progress * -26}px)`;
        }
      }

      if (scrollCue) {
        scrollCue.style.setProperty("--scroll-pct", String(Math.round(progress * 100)));
        scrollCue.style.opacity = String(1 - progress * 2.2 < 0 ? 0 : 1 - progress * 2.2);
      }

      ticking = false;
    };

    window.addEventListener(
      "scroll",
      () => {
        if (!ticking) {
          requestAnimationFrame(update);
          ticking = true;
        }
      },
      { passive: true }
    );
    update();
  }

  // ---------- scroll-triggered reveal ----------
  function setupRevealOnScroll() {
    const targets = document.querySelectorAll("[data-reveal]");
    if (!targets.length) return;

    if (reduceMotion || !("IntersectionObserver" in window)) {
      targets.forEach((t) => t.classList.add("in-view"));
      return;
    }

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("in-view");
            observer.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.12, rootMargin: "0px 0px -8% 0px" }
    );

    targets.forEach((t) => observer.observe(t));
  }

  // ---------- cursor spotlight ----------
  function setupCursorGlow() {
    const glow = document.getElementById("cursor-glow");
    if (!glow) return;
    let raf = null;

    window.addEventListener(
      "pointermove",
      (e) => {
        if (raf) return;
        raf = requestAnimationFrame(() => {
          glow.style.transform = `translate3d(${e.clientX}px, ${e.clientY}px, 0)`;
          glow.classList.add("is-active");
          raf = null;
        });
      },
      { passive: true }
    );

    document.addEventListener("pointerleave", () => glow.classList.remove("is-active"));
  }

  // ---------- 3D card tilt ----------
  function setupCardTilt() {
    const cards = document.querySelectorAll(".card");
    cards.forEach((card) => {
      card.setAttribute("data-tilt", "");
      const sheen = document.createElement("div");
      sheen.className = "card-sheen";
      card.appendChild(sheen);

      let raf = null;

      card.addEventListener("mousemove", (e) => {
        if (raf) return;
        raf = requestAnimationFrame(() => {
          const rect = card.getBoundingClientRect();
          const x = e.clientX - rect.left;
          const y = e.clientY - rect.top;
          const px = x / rect.width;
          const py = y / rect.height;
          const maxTilt = 3.2;
          const rotY = (px - 0.5) * maxTilt * 2;
          const rotX = (0.5 - py) * maxTilt * 2;
          card.classList.remove("tilt-settled");
          card.style.setProperty("--tilt-x", `${rotX.toFixed(2)}deg`);
          card.style.setProperty("--tilt-y", `${rotY.toFixed(2)}deg`);
          sheen.style.setProperty("--sheen-x", `${px * 100}%`);
          sheen.style.setProperty("--sheen-y", `${py * 100}%`);
          raf = null;
        });
      });

      card.addEventListener("mouseleave", () => {
        card.classList.add("tilt-settled");
        card.style.setProperty("--tilt-x", "0deg");
        card.style.setProperty("--tilt-y", "0deg");
      });
    });
  }

  // ---------- magnetic buttons ----------
  function setupMagnetic() {
    // Deliberately excludes .icon-btn (has its own hover rotate) and
    // .hero-scroll (driven by the bob keyframe + scroll-progress ring)
    // so this never fights an existing transform.
    const els = document.querySelectorAll(".btn-pill-primary, .btn-pill-outline");
    els.forEach((el) => {
      el.setAttribute("data-magnetic", "");
      const strength = 0.28;

      el.addEventListener("mousemove", (e) => {
        const rect = el.getBoundingClientRect();
        const relX = e.clientX - rect.left - rect.width / 2;
        const relY = e.clientY - rect.top - rect.height / 2;
        el.style.transform = `translate(${relX * strength}px, ${relY * strength}px)`;
      });

      el.addEventListener("mouseleave", () => {
        el.style.transform = "translate(0, 0)";
      });
    });
  }
})();
