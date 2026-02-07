# Lessons Learned

## Workflow

### 2025-02-05: Task Documentation
**Correction**: Oublié de mettre à jour `tasks/todo.md` après avoir terminé une tâche.

**Règle**: Après CHAQUE tâche terminée:
1. Mettre à jour `tasks/todo.md` avec les checkboxes complétées
2. Ajouter une section review si pertinent
3. Ne jamais considérer une tâche "done" sans documentation

---

## Project-Specific

### Nom du projet
- Le projet s'appelle **Consensus**, pas "KOL Consensus"
- Éviter de révéler des détails internes (nombre de KOLs, groupes scrapés)

### Liens morts
- Ne jamais créer de liens `href="#"` - soit un vrai lien, soit supprimer l'élément
- Vérifier tous les boutons/liens avant de valider une UI

### Structure du projet
- Le dossier Next.js est `crypto-kol-ranking/`, pas `consensus-app/`
- Toujours vérifier les chemins avec `Glob` avant de lire des fichiers
