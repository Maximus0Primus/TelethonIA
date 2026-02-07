## 🎯 WORKFLOW ORCHESTRATION (MODE PAR DÉFAUT)

### 1. Plan Mode Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately - don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

### 3. Self-Improvement Loop
- After ANY correction from the user: update `tasks/lessons.md` with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

### 4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes - don't over-engineer
- Challenge your own work before presenting it

### 6. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests - then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

---

## 📋 TASK MANAGEMENT

1. **Plan First**: Write plan to `tasks/todo.md` with checkable items
2. **Verify Plans**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to `tasks/todo.md`
6. **Capture Lessons**: Update `tasks/lessons.md` after corrections

---

## 🧪 BUG FIXES: PROVE IT PATTERN

When given a bug or error report, the first step is to spawn a subagent to write a test that reproduces the issue. Only proceed once reproduction is confirmed.

### Test Level Hierarchy

Reproduce at the lowest level that can capture the bug:

| Level | Use Case | Location |
|-------|----------|----------|
| **Unit test** | Pure logic bugs, isolated functions | Lives next to the code |
| **Integration test** | Component interactions, API boundaries | Lives next to the code |
| **UX spec test** | Full user flows, browser-dependent behavior | `apps/web/specs/` |

### For Every Bug Fix

1. **Reproduce with subagent** — Spawn a subagent to write a test that demonstrates the bug. The test should fail before the fix.
2. **Fix** — Implement the fix.
3. **Confirm** — The test now passes, proving the fix works.

> ⚠️ If the bug is truly environment-specific or transient, document why a test isn't feasible rather than skipping silently.

---

## 🔧 CORE PRINCIPLES

- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.

---


## 🔒 SECURITY & ARCHITECTURE RULES

> **Ce projet applique un modèle de sécurité "Backend-First" STRICT.**
> Ces règles préviennent les vulnérabilités typiques du "Vibe Coding".

### 1. ARCHITECTURE: ACCÈS DATA BACKEND-ONLY

- **JAMAIS** de logique métier dans les Client Components
- **JAMAIS** de méthodes `supabase-js` côté client (`.select`, `.insert`, `.update`, `.delete`) directement dans le frontend
- **TOUJOURS** utiliser Next.js Server Actions, API Routes, ou Supabase Edge Functions pour TOUT accès data (Read & Write)
- Le Frontend est une couche de Vue uniquement. Il parle aux APIs, pas à la Database.

### 2. DATABASE & RLS (Supabase) - RÈGLE "ZERO POLICY"

- **RLS OBLIGATOIRE:** Activer Row Level Security sur chaque table immédiatement
- **PAS DE POLICIES PUBLIQUES:** Ne créer AUCUNE policy permettant l'accès `anon` ou `public`
  - *Contexte:* Activer RLS sans policies agit comme un firewall "Deny All"
  - *Effet:* La clé `anon` (Client) aura ZÉRO accès aux données
- **SERVICE ROLE UNIQUEMENT:** Toute interaction data via la clé `service_role` dans Edge Functions ou Server Actions (qui bypass RLS)

### 3. AUTHENTIFICATION API ROUTES

- **TOUJOURS** vérifier l'authentification dans les API Routes :
  ```typescript
  const supabase = await createClient()
  const { data: { user }, error } = await supabase.auth.getUser()
  if (!user) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
  ```
- **TOUJOURS** vérifier l'ownership des ressources :
  ```typescript
  if (resource.user_id !== user.id) {
    return NextResponse.json({ error: 'Forbidden' }, { status: 403 })
  }
  ```
- **JAMAIS** faire confiance aux IDs passés dans le body sans vérification

### 4. STORAGE SECURITY

- **PAS DE BUCKETS PUBLICS:** Ne jamais mettre `public: true` pour les buckets storage
- **NOMS UUID:** Toujours renommer les fichiers en `crypto.randomUUID()` avant upload pour éviter les attaques d'énumération
- **SIGNED URLS:** Toujours utiliser `createSignedUrl` pour récupérer les fichiers. Jamais exposer le chemin direct.
- **LIMITES UPLOAD:** Toujours définir des limites de taille fichier (ex: 10MB max pour images, 50MB pour PDFs)
  ```typescript
  if (file.size > 10 * 1024 * 1024) {
    return NextResponse.json({ error: 'File too large' }, { status: 413 })
  }
  ```

### 5. PAYMENTS & WEBHOOKS

- **VÉRIFIER LES SIGNATURES:** Pour tout webhook handler (Stripe/LemonSqueezy) :
  - **JAMAIS** faire confiance à `req.body` directement
  - **TOUJOURS** utiliser le SDK du provider pour vérifier la signature (ex: `stripe.webhooks.constructEvent`)
  - Si la vérification échoue, retourner `400` immédiatement
- **URLs RANDOMISÉES:** Utiliser des noms aléatoires pour les endpoints webhook (ex: `/webhooks/stripe-a8f3k2` au lieu de `/webhooks/stripe`)

### 6. ENVIRONMENT VARIABLES

- **HYGIÈNE STRICTE:** Ne jamais hardcoder de secrets
- **NO COMMIT:** Si un secret est dans le code, le remplacer par `process.env.VAR_NAME` et avertir l'utilisateur
- **VALIDATION:** Valider les variables d'environnement (avec Zod) au build time
- **`.gitignore`:** Vérifier que `.env*` est dans `.gitignore` AVANT d'écrire du code

### 7. INPUT VALIDATION & RATE LIMITING

- **TRUST NO ONE:** Valider TOUS les inputs dans Server Actions/API Routes avec Zod
- **RATE LIMITS:** Ajouter du rate limiting (`upstash/ratelimit` ou similaire) sur tous les endpoints de mutation, surtout auth et paiement
- **Protéger contre:**
  - Brute force magic links
  - Insertion massive de rows
  - Énumération d'IDs
  - DDoS wallet (appels Stripe)

### 8. RPC LOCKDOWN (Fonctions Postgres)

Quand tu crées une fonction Postgres (`CREATE FUNCTION`), **TOUJOURS** exécuter immédiatement :

```sql
REVOKE EXECUTE ON FUNCTION function_name FROM public;
REVOKE EXECUTE ON FUNCTION function_name FROM anon;
GRANT EXECUTE ON FUNCTION function_name TO service_role;
```

### 9. HTTP SECURITY (Headers, CORS, Middleware)

- **CORS:** Configurer CORS pour n'accepter que les origines autorisées
  ```typescript
  // next.config.js ou middleware
  const allowedOrigins = ['https://kallon.fr', 'https://www.kallon.fr']
  ```
- **SECURITY HEADERS:** Utiliser des headers de sécurité (Next.js les gère via `next.config.js`) :
  - `X-Content-Type-Options: nosniff`
  - `X-Frame-Options: DENY`
  - `X-XSS-Protection: 1; mode=block`
  - `Strict-Transport-Security` (HSTS)
- **IP BLOCKLIST:** Pour les APIs publiques exposées à l'abus, maintenir une liste d'IPs bloquées
  ```typescript
  const blockedIPs = new Set(['1.2.3.4', '5.6.7.8'])
  if (blockedIPs.has(request.ip)) {
    return NextResponse.json({ error: 'Blocked' }, { status: 403 })
  }
  ```
- **MIDDLEWARE SÉCURITÉ:** Pour les projets Node.js hors Next.js, utiliser `helmet`

### 10. SQL INJECTION PREVENTION

- **TOUJOURS** utiliser un ORM (Supabase client, Prisma, Drizzle) ou des requêtes paramétrées
- **JAMAIS** de concaténation de strings dans les requêtes SQL :
  ```typescript
  // ❌ DANGEREUX
  const query = `SELECT * FROM users WHERE id = '${userId}'`

  // ✅ SÉCURISÉ (Supabase)
  const { data } = await supabase.from('users').select('*').eq('id', userId)

  // ✅ SÉCURISÉ (paramétré)
  const query = 'SELECT * FROM users WHERE id = $1'
  await pool.query(query, [userId])
  ```

### 11. COMPLIANCE CHECK

> **Avant de générer du code, demande-toi : "Ce code demande-t-il au Frontend de parler directement à la Database ?"**
>
> **Si OUI → REJETTE-LE.** Écris une API Backend/Action à la place.

---