#lang racket
(define (find_closest_elements numbers)
  (define sorted (sort numbers <))
  (let loop ([lst sorted] [best #f] [bestd +inf.0])
    (cond [(null? (cdr lst)) best]
          [else (define d (- (cadr lst) (car lst)))
                (if (< d bestd)
                    (loop (cdr lst) (list (car lst) (cadr lst)) d)
                    (loop (cdr lst) best bestd))])))
