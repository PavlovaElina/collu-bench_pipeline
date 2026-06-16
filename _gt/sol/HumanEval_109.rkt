#lang racket
(define (move_one_ball arr)
  (cond [(null? arr) #t]
        [else (define s (sort arr <))
              (define n (length arr))
              (for/or ([k (in-range n)])
                (equal? (append (drop arr k) (take arr k)) s))]))
