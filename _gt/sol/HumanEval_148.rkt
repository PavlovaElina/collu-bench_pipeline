#lang racket
(define (bf planet1 planet2)
  (define planets (vector "Mercury" "Venus" "Earth" "Mars" "Jupiter" "Saturn" "Uranus" "Neptune"))
  (define (idx p) (for/first ([i (in-range 8)] #:when (string=? (vector-ref planets i) p)) i))
  (define i1 (idx planet1)) (define i2 (idx planet2))
  (cond [(or (not i1) (not i2)) '()]
        [else (define lo (min i1 i2)) (define hi (max i1 i2))
              (for/list ([i (in-range (+ lo 1) hi)]) (vector-ref planets i))]))
